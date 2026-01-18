import numpy as np
from scipy import ndimage as ndi
from ._geometric import SimilarityTransform, AffineTransform, ProjectiveTransform
from ._warps_cy import _warp_fast
from ..measure import block_reduce
from .._shared.utils import (
def _local_mean_weights(old_size, new_size, grid_mode, dtype):
    """Create a 2D weight matrix for resizing with the local mean.

    Parameters
    ----------
    old_size: int
        Old size.
    new_size: int
        New size.
    grid_mode : bool
        Whether to use grid data model of pixel/voxel model for
        average weights computation.
    dtype: dtype
        Output array data type.

    Returns
    -------
    weights: (new_size, old_size) array
        Rows sum to 1.

    """
    if grid_mode:
        old_breaks = np.linspace(0, old_size, num=old_size + 1, dtype=dtype)
        new_breaks = np.linspace(0, old_size, num=new_size + 1, dtype=dtype)
    else:
        old, new = (old_size - 1, new_size - 1)
        old_breaks = np.pad(np.linspace(0.5, old - 0.5, old, dtype=dtype), 1, 'constant', constant_values=(0, old))
        if new == 0:
            val = np.inf
        else:
            val = 0.5 * old / new
        new_breaks = np.pad(np.linspace(val, old - val, new, dtype=dtype), 1, 'constant', constant_values=(0, old))
    upper = np.minimum(new_breaks[1:, np.newaxis], old_breaks[np.newaxis, 1:])
    lower = np.maximum(new_breaks[:-1, np.newaxis], old_breaks[np.newaxis, :-1])
    weights = np.maximum(upper - lower, 0)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights