import functools
import warnings
import numpy as np
from numpy.lib import function_base
from numpy.core import overrides
def _nanquantile_1d(arr1d, q, overwrite_input=False, method='linear'):
    """
    Private function for rank 1 arrays. Compute quantile ignoring NaNs.
    See nanpercentile for parameter usage
    """
    arr1d, overwrite_input = _remove_nan_1d(arr1d, overwrite_input=overwrite_input)
    if arr1d.size == 0:
        return np.full(q.shape, np.nan, dtype=arr1d.dtype)[()]
    return function_base._quantile_unchecked(arr1d, q, overwrite_input=overwrite_input, method=method)