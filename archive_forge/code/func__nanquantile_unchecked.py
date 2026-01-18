import functools
import warnings
import numpy as np
from numpy.lib import function_base
from numpy.core import overrides
def _nanquantile_unchecked(a, q, axis=None, out=None, overwrite_input=False, method='linear', keepdims=np._NoValue):
    """Assumes that q is in [0, 1], and is an ndarray"""
    if a.size == 0:
        return np.nanmean(a, axis, out=out, keepdims=keepdims)
    return function_base._ureduce(a, func=_nanquantile_ureduce_func, q=q, keepdims=keepdims, axis=axis, out=out, overwrite_input=overwrite_input, method=method)