from typing import *
import cv2
import numpy as np
from skimage.measure import block_reduce

def min_pool(im: np.ndarray) -> np.ndarray:
    """Take the minimum of each 2x2 region of pixels, returning an image half the size of the original (ceil'd)."""
     # TODO(wrsturgeon): randomize L/R and U/D bias (pool on the left/right and top/bottom equally)
    return block_reduce(im, block_size=(2, 2), func=np.min, cval=256)

class WassersteinPyramid:
    """
    Data structure for rapid lookup of the spatially closest pixel arbitrarily close in color to `key` for any queried pixel.
    Uses L1 (Manhattan) distance between BGR values.
    """
    def __init__(self, image_path: str, key: Union[np.uint8, np.ndarray] = 0) -> None:
        """
        Initialize a Wasserstein pyramid for the image found at `image_path` for looking up colors close to `key`.
        `key` takes either a scalar (grayscale) or a 3-tuple (BGR).
        """
        layer = cv2.imread(image_path)
        self.shape = layer.shape
        assert layer is not None, f"Couldn't read the image ostensibly at {image_path}"
        assert layer.dtype == np.uint8, f"Image at {image_path} must be encoded as unsigned 8-bit values (np.uint8), but got {layer.dtype}"
        assert np.isscalar(key) or len(key) == layer.shape[-1], f"`key` must be a scalar or a vector matching `image_path`'s channels ({layer.shape[-1]}), but got {key}"
        assert np.all(np.asarray(key) >= 0) and np.all(np.asarray(key) <= 255), f"`key` must be in the range [0, 255], but got {key}"
        layer = np.abs(layer.astype(np.int16) - key).astype(np.uint16)
        layer = np.sum(layer, axis=-1, dtype=np.uint16)
        self.layers = [layer] # first layer := absolute difference btn each pixel and `key` (16b to prevent overflow)
        while self.layers[-1].shape[0] > 1 or self.layers[-1].shape[1] > 1:
            self.layers.append(min_pool(self.layers[-1]))
        self.key = key
    def __getitem__(self, index: int) -> np.ndarray:
        """Return the layer at `index`, largest first."""
        return self.layers[index]
    def __len__(self) -> int:
        """Number of layers in the pyramid."""
        return len(self.layers)
    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over the layers in the pyramid, largest first."""
        return iter(self.layers)
    def __reversed__(self) -> Iterator[np.ndarray]:
        """Iterate over the layers in the pyramid, smallest first."""
        return reversed(self.layers)
    def __repr__(self) -> str:
        """Print layer sizes, largest first."""
        return f"WassersteinPyramid[{', '.join(str(layer.shape) for layer in self)}]"
    def __str__(self) -> str:
        """Print layer sizes, largest first."""
        return self.__repr__()
    def __call__(self,
                query: Tuple[np.uint16, np.uint16],
                tolerance: np.uint16 = 0,
                precise: bool = False
            ) -> Tuple[np.int16, np.int16]:
        """
        Spatial *distance* from `query` to the closest pixel at most `threshold` BGR values away from `key` (specified at initialization).
        Note that `query` takes (y, x) values (not (x, y)); this function returns (y, x) values as well.
        """
        y, x = query
        assert y >= 0 and y < self.shape[0], f"query y-index out of range: takes [0, {self.shape[0]}), but got {y}"
        assert x >= 0 and x < self.shape[1], f"query x-index out of range: takes [0, {self.shape[1]}), but got {x}"
        assert tolerance >= 0, f"tolerance must be nonnegative, but got {tolerance}"
        for idx, layer in enumerate(self): # Non-corners first, then corners
            #  [                bounds checking                ] and [         <= tolerance         ]: return ...
            if y >                  0                            and layer[y - 1, x    ] <= tolerance: return self._find_precise(idx, (y - 1, x    ), tolerance, query) if precise else (-1 << idx,  0       )
            if y < layer.shape[0] - 1                            and layer[y + 1, x    ] <= tolerance: return self._find_precise(idx, (y + 1, x    ), tolerance, query) if precise else ( 1 << idx,  0       )
            if                            x >                  0 and layer[y    , x - 1] <= tolerance: return self._find_precise(idx, (y    , x - 1), tolerance, query) if precise else ( 0       , -1 << idx)
            if                            x < layer.shape[1] - 1 and layer[y    , x + 1] <= tolerance: return self._find_precise(idx, (y    , x + 1), tolerance, query) if precise else ( 0       ,  1 << idx)
            if y >                  0 and x >                  0 and layer[y - 1, x - 1] <= tolerance: return self._find_precise(idx, (y - 1, x - 1), tolerance, query) if precise else (-1 << idx, -1 << idx)
            if y >                  0 and x < layer.shape[1] - 1 and layer[y - 1, x + 1] <= tolerance: return self._find_precise(idx, (y - 1, x + 1), tolerance, query) if precise else (-1 << idx,  1 << idx)
            if y < layer.shape[0] - 1 and x >                  0 and layer[y + 1, x - 1] <= tolerance: return self._find_precise(idx, (y + 1, x - 1), tolerance, query) if precise else ( 1 << idx, -1 << idx)
            if y < layer.shape[0] - 1 and x < layer.shape[1] - 1 and layer[y + 1, x + 1] <= tolerance: return self._find_precise(idx, (y + 1, x + 1), tolerance, query) if precise else ( 1 << idx,  1 << idx)
            # "Downsample" coordinates
            y >>= 1
            x >>= 1
        raise ValueError(f"No pixel within tolerance {tolerance} of {self.key} found in the image")
    def _find_precise(self,
                layer_idx: np.uint8,
                layer_pos: Tuple[np.uint16, np.uint16],
                tolerance: np.uint16,
                original_query: Tuple[np.uint16, np.uint16]
            ) -> Tuple[np.int16, np.int16]:
        """Given the imprecise output of __call__, finds the precise pixel that caused the min-pooled sub-tolerance value."""
        # TODO(wrsturgeon): accept dy & dx values for tiebreaking
        # no need for a 3x3 like in the upward case--just 2D binary search
        assert layer_idx < len(self), f"layer_idx out of range: takes [0, {len(self)}), but got {layer_idx}"
        y, x = layer_pos
        assert y >= 0 and y < self[layer_idx].shape[0], f"layer_pos y-index out of range: takes [0, {self[layer_idx].shape[0]}), but got {y}"
        assert x >= 0 and x < self[layer_idx].shape[1], f"layer_pos x-index out of range: takes [0, {self[layer_idx].shape[1]}), but got {x}"
        assert tolerance >= 0, f"tolerance must be nonnegative, but got {tolerance}"
        for idx, layer in enumerate(reversed(self[:layer_idx])): # layer_idx - 1, layer_idx - 2, ..., 0
            y <<= 1
            x <<= 1
            if                              layer[y    , x    ] <= tolerance: continue
            if (x | 1) < layer.shape[1] and layer[y    , x | 1] <= tolerance: x |= 1; continue
            if (y | 1) < layer.shape[0] and layer[y | 1, x    ] <= tolerance: y |= 1; continue
            if                              layer[y | 1, x | 1] <= tolerance: y |= 1; x |= 1; continue
            raise ValueError(f"Internal error: no pixel in layer {idx} <= tolerance ({tolerance}), but imprecise search found one")
        return (y - original_query[0], x - original_query[1])

# Test
if __name__ == "__main__":

    # Arguments
    image_path = "images/blurred.png"
    query = (50, 350)
    key = (0, 0, 255)
    tolerance = 96
    precise = False
    
    # Evaluation
    pyr = WassersteinPyramid(image_path, key)
    print(pyr)
    dst = pyr(query, tolerance, precise)
    pos = (query[0] + dst[0], query[1] + dst[1])
    euclidean = np.sqrt(np.sum(np.square(np.asarray(dst))))
    im = cv2.arrowedLine(cv2.imread(image_path), (query[1], query[0]), (pos[1], pos[0]), color=(0, 0, 255), thickness=int(0.02 * euclidean), tipLength=0.2)
    cv2.imshow(f"Distance to BGR={key} +/- {tolerance} from (y, x)={query}", im)
    cv2.waitKey(0)
