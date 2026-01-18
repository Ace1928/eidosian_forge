from __future__ import annotations
import operator
from operator import (
import textwrap
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.interval import (
from pandas._libs.missing import NA
from pandas._typing import (
from pandas.compat.numpy import function as nv
from pandas.errors import IntCastingNaNError
from pandas.util._decorators import Appender
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core.algorithms import (
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.arrays.base import (
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.arrays.timedeltas import TimedeltaArray
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.indexers import check_array_indexer
from pandas.core.ops import (
from_arrays
from_tuples
from_breaks
@classmethod
def _ensure_simple_new_inputs(cls, left, right, closed: IntervalClosedType | None=None, copy: bool=False, dtype: Dtype | None=None) -> tuple[IntervalSide, IntervalSide, IntervalDtype]:
    """Ensure correctness of input parameters for cls._simple_new."""
    from pandas.core.indexes.base import ensure_index
    left = ensure_index(left, copy=copy)
    left = maybe_upcast_numeric_to_64bit(left)
    right = ensure_index(right, copy=copy)
    right = maybe_upcast_numeric_to_64bit(right)
    if closed is None and isinstance(dtype, IntervalDtype):
        closed = dtype.closed
    closed = closed or 'right'
    if dtype is not None:
        dtype = pandas_dtype(dtype)
        if isinstance(dtype, IntervalDtype):
            if dtype.subtype is not None:
                left = left.astype(dtype.subtype)
                right = right.astype(dtype.subtype)
        else:
            msg = f'dtype must be an IntervalDtype, got {dtype}'
            raise TypeError(msg)
        if dtype.closed is None:
            dtype = IntervalDtype(dtype.subtype, closed)
        elif closed != dtype.closed:
            raise ValueError('closed keyword does not match dtype.closed')
    if is_float_dtype(left.dtype) and is_integer_dtype(right.dtype):
        right = right.astype(left.dtype)
    elif is_float_dtype(right.dtype) and is_integer_dtype(left.dtype):
        left = left.astype(right.dtype)
    if type(left) != type(right):
        msg = f'must not have differing left [{type(left).__name__}] and right [{type(right).__name__}] types'
        raise ValueError(msg)
    if isinstance(left.dtype, CategoricalDtype) or is_string_dtype(left.dtype):
        msg = 'category, object, and string subtypes are not supported for IntervalArray'
        raise TypeError(msg)
    if isinstance(left, ABCPeriodIndex):
        msg = 'Period dtypes are not supported, use a PeriodIndex instead'
        raise ValueError(msg)
    if isinstance(left, ABCDatetimeIndex) and str(left.tz) != str(right.tz):
        msg = f"left and right must have the same time zone, got '{left.tz}' and '{right.tz}'"
        raise ValueError(msg)
    elif needs_i8_conversion(left.dtype) and left.unit != right.unit:
        left_arr, right_arr = left._data._ensure_matching_resos(right._data)
        left = ensure_index(left_arr)
        right = ensure_index(right_arr)
    left = ensure_wrapped_if_datetimelike(left)
    left = extract_array(left, extract_numpy=True)
    right = ensure_wrapped_if_datetimelike(right)
    right = extract_array(right, extract_numpy=True)
    if isinstance(left, ArrowExtensionArray) or isinstance(right, ArrowExtensionArray):
        pass
    else:
        lbase = getattr(left, '_ndarray', left)
        lbase = getattr(lbase, '_data', lbase).base
        rbase = getattr(right, '_ndarray', right)
        rbase = getattr(rbase, '_data', rbase).base
        if lbase is not None and lbase is rbase:
            right = right.copy()
    dtype = IntervalDtype(left.dtype, closed=closed)
    return (left, right, dtype)