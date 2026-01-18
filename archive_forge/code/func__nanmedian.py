import functools
import warnings
import numpy as np
from numpy.lib import function_base
from numpy.core import overrides
def _nanmedian(a, axis=None, out=None, overwrite_input=False):
    """
    Private function that doesn't support extended axis or keepdims.
    These methods are extended to this function using _ureduce
    See nanmedian for parameter usage

    """
    if axis is None or a.ndim == 1:
        part = a.ravel()
        if out is None:
            return _nanmedian1d(part, overwrite_input)
        else:
            out[...] = _nanmedian1d(part, overwrite_input)
            return out
    else:
        if a.shape[axis] < 600:
            return _nanmedian_small(a, axis, out, overwrite_input)
        result = np.apply_along_axis(_nanmedian1d, axis, a, overwrite_input)
        if out is not None:
            out[...] = result
        return result