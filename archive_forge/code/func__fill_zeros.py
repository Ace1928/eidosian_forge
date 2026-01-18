from numpy in the following ways:
from __future__ import annotations
import operator
import numpy as np
from pandas.core import roperator
def _fill_zeros(result: np.ndarray, x, y):
    """
    If this is a reversed op, then flip x,y

    If we have an integer value (or array in y)
    and we have 0's, fill them with np.nan,
    return the result.

    Mask the nan's from x.
    """
    if result.dtype.kind == 'f':
        return result
    is_variable_type = hasattr(y, 'dtype')
    is_scalar_type = not isinstance(y, np.ndarray)
    if not is_variable_type and (not is_scalar_type):
        return result
    if is_scalar_type:
        y = np.array(y)
    if y.dtype.kind in 'iu':
        ymask = y == 0
        if ymask.any():
            mask = ymask & ~np.isnan(result)
            result = result.astype('float64', copy=False)
            np.putmask(result, mask, np.nan)
    return result