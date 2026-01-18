from __future__ import annotations
import datetime as dt
import functools
from typing import (
import warnings
import numpy as np
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import (
from pandas._libs.missing import (
from pandas._libs.tslibs import (
from pandas._libs.tslibs.timedeltas import array_to_timedelta64
from pandas.compat.numpy import np_version_gt2
from pandas.errors import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_list_like
from pandas.core.dtypes.missing import (
from pandas.io._util import _arrow_dtype_mapping
def _maybe_promote(dtype: np.dtype, fill_value=np.nan):
    if not is_scalar(fill_value):
        if dtype != object:
            raise ValueError('fill_value must be a scalar')
        dtype = _dtype_obj
        return (dtype, fill_value)
    if is_valid_na_for_dtype(fill_value, dtype) and dtype.kind in 'iufcmM':
        dtype = ensure_dtype_can_hold_na(dtype)
        fv = na_value_for_dtype(dtype)
        return (dtype, fv)
    elif isinstance(dtype, CategoricalDtype):
        if fill_value in dtype.categories or isna(fill_value):
            return (dtype, fill_value)
        else:
            return (object, ensure_object(fill_value))
    elif isna(fill_value):
        dtype = _dtype_obj
        if fill_value is None:
            fill_value = np.nan
        return (dtype, fill_value)
    if issubclass(dtype.type, np.datetime64):
        inferred, fv = infer_dtype_from_scalar(fill_value)
        if inferred == dtype:
            return (dtype, fv)
        from pandas.core.arrays import DatetimeArray
        dta = DatetimeArray._from_sequence([], dtype='M8[ns]')
        try:
            fv = dta._validate_setitem_value(fill_value)
            return (dta.dtype, fv)
        except (ValueError, TypeError):
            return (_dtype_obj, fill_value)
    elif issubclass(dtype.type, np.timedelta64):
        inferred, fv = infer_dtype_from_scalar(fill_value)
        if inferred == dtype:
            return (dtype, fv)
        elif inferred.kind == 'm':
            unit = np.datetime_data(dtype)[0]
            try:
                td = Timedelta(fill_value).as_unit(unit, round_ok=False)
            except OutOfBoundsTimedelta:
                return (_dtype_obj, fill_value)
            else:
                return (dtype, td.asm8)
        return (_dtype_obj, fill_value)
    elif is_float(fill_value):
        if issubclass(dtype.type, np.bool_):
            dtype = np.dtype(np.object_)
        elif issubclass(dtype.type, np.integer):
            dtype = np.dtype(np.float64)
        elif dtype.kind == 'f':
            mst = np.min_scalar_type(fill_value)
            if mst > dtype:
                dtype = mst
        elif dtype.kind == 'c':
            mst = np.min_scalar_type(fill_value)
            dtype = np.promote_types(dtype, mst)
    elif is_bool(fill_value):
        if not issubclass(dtype.type, np.bool_):
            dtype = np.dtype(np.object_)
    elif is_integer(fill_value):
        if issubclass(dtype.type, np.bool_):
            dtype = np.dtype(np.object_)
        elif issubclass(dtype.type, np.integer):
            if not np_can_cast_scalar(fill_value, dtype):
                mst = np.min_scalar_type(fill_value)
                dtype = np.promote_types(dtype, mst)
                if dtype.kind == 'f':
                    dtype = np.dtype(np.object_)
    elif is_complex(fill_value):
        if issubclass(dtype.type, np.bool_):
            dtype = np.dtype(np.object_)
        elif issubclass(dtype.type, (np.integer, np.floating)):
            mst = np.min_scalar_type(fill_value)
            dtype = np.promote_types(dtype, mst)
        elif dtype.kind == 'c':
            mst = np.min_scalar_type(fill_value)
            if mst > dtype:
                dtype = mst
    else:
        dtype = np.dtype(np.object_)
    if issubclass(dtype.type, (bytes, str)):
        dtype = np.dtype(np.object_)
    fill_value = _ensure_dtype_type(fill_value, dtype)
    return (dtype, fill_value)