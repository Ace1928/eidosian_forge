from __future__ import annotations
import functools
from typing import (
import numpy as np
from pandas._libs import (
from pandas.core.dtypes.cast import maybe_promote
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import na_value_for_dtype
from pandas.core.construction import ensure_wrapped_if_datetimelike
def _take_nd_ndarray(arr: np.ndarray, indexer: npt.NDArray[np.intp] | None, axis: AxisInt, fill_value, allow_fill: bool) -> np.ndarray:
    if indexer is None:
        indexer = np.arange(arr.shape[axis], dtype=np.intp)
        dtype, fill_value = (arr.dtype, arr.dtype.type())
    else:
        indexer = ensure_platform_int(indexer)
    dtype, fill_value, mask_info = _take_preprocess_indexer_and_fill_value(arr, indexer, fill_value, allow_fill)
    flip_order = False
    if arr.ndim == 2 and arr.flags.f_contiguous:
        flip_order = True
    if flip_order:
        arr = arr.T
        axis = arr.ndim - axis - 1
    out_shape_ = list(arr.shape)
    out_shape_[axis] = len(indexer)
    out_shape = tuple(out_shape_)
    if arr.flags.f_contiguous and axis == arr.ndim - 1:
        out = np.empty(out_shape, dtype=dtype, order='F')
    else:
        out = np.empty(out_shape, dtype=dtype)
    func = _get_take_nd_function(arr.ndim, arr.dtype, out.dtype, axis=axis, mask_info=mask_info)
    func(arr, indexer, out, fill_value)
    if flip_order:
        out = out.T
    return out