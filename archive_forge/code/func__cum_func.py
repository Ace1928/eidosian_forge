from __future__ import annotations
from typing import (
import numpy as np
def _cum_func(func: Callable, values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool=True):
    """
    Accumulations for 1D masked array.

    We will modify values in place to replace NAs with the appropriate fill value.

    Parameters
    ----------
    func : np.cumsum, np.cumprod, np.maximum.accumulate, np.minimum.accumulate
    values : np.ndarray
        Numpy array with the values (can be of any dtype that support the
        operation).
    mask : np.ndarray
        Boolean numpy array (True values indicate missing values).
    skipna : bool, default True
        Whether to skip NA.
    """
    dtype_info: np.iinfo | np.finfo
    if values.dtype.kind == 'f':
        dtype_info = np.finfo(values.dtype.type)
    elif values.dtype.kind in 'iu':
        dtype_info = np.iinfo(values.dtype.type)
    elif values.dtype.kind == 'b':
        dtype_info = np.iinfo(np.uint8)
    else:
        raise NotImplementedError(f'No masked accumulation defined for dtype {values.dtype.type}')
    try:
        fill_value = {np.cumprod: 1, np.maximum.accumulate: dtype_info.min, np.cumsum: 0, np.minimum.accumulate: dtype_info.max}[func]
    except KeyError:
        raise NotImplementedError(f'No accumulation for {func} implemented on BaseMaskedArray')
    values[mask] = fill_value
    if not skipna:
        mask = np.maximum.accumulate(mask)
    values = func(values)
    return (values, mask)