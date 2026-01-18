from warnings import warn
import numpy as np
from scipy import linalg
from .._shared.utils import (
from ..util import dtype, dtype_limits
def _prepare_lab_array(arr, force_copy=True):
    """Ensure input for lab2lch and lch2lab is well-formed.

    Input array must be in floating point and have at least 3 elements in the
    last dimension. Returns a new array by default.
    """
    arr = np.asarray(arr)
    shape = arr.shape
    if shape[-1] < 3:
        raise ValueError('Input image has less than 3 channels.')
    float_dtype = _supported_float_type(arr.dtype)
    if float_dtype == np.float32:
        _func = dtype.img_as_float32
    else:
        _func = dtype.img_as_float64
    return _func(arr, force_copy=force_copy)