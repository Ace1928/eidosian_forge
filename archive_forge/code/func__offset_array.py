import numpy as np
from ..util.dtype import dtype_range, dtype_limits
from .._shared import utils
def _offset_array(arr, low_boundary, high_boundary):
    """Offset the array to get the lowest value at 0 if negative."""
    if low_boundary < 0:
        offset = low_boundary
        dyn_range = high_boundary - low_boundary
        offset_dtype = np.promote_types(np.min_scalar_type(dyn_range), np.min_scalar_type(low_boundary))
        if arr.dtype != offset_dtype:
            arr = arr.astype(offset_dtype)
        arr = arr - offset
    return arr