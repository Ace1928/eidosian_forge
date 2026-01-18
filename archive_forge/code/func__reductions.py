from __future__ import annotations
from typing import (
import warnings
import numpy as np
from pandas._libs import missing as libmissing
from pandas.core.nanops import check_below_min_count
def _reductions(func: Callable, values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool=True, min_count: int=0, axis: AxisInt | None=None, **kwargs):
    """
    Sum, mean or product for 1D masked array.

    Parameters
    ----------
    func : np.sum or np.prod
    values : np.ndarray
        Numpy array with the values (can be of any dtype that support the
        operation).
    mask : np.ndarray[bool]
        Boolean numpy array (True values indicate missing values).
    skipna : bool, default True
        Whether to skip NA.
    min_count : int, default 0
        The required number of valid values to perform the operation. If fewer than
        ``min_count`` non-NA values are present the result will be NA.
    axis : int, optional, default None
    """
    if not skipna:
        if mask.any() or check_below_min_count(values.shape, None, min_count):
            return libmissing.NA
        else:
            return func(values, axis=axis, **kwargs)
    else:
        if check_below_min_count(values.shape, mask, min_count) and (axis is None or values.ndim == 1):
            return libmissing.NA
        return func(values, where=~mask, axis=axis, **kwargs)