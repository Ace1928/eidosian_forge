from __future__ import annotations
from abc import (
from contextlib import (
from datetime import (
from functools import partial
import re
from typing import (
import warnings
import numpy as np
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
from pandas import get_option
from pandas.core.api import (
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.common import maybe_make_list
from pandas.core.internals.construction import convert_object_array
from pandas.core.tools.datetimes import to_datetime
def _sqlalchemy_type(self, col: Index | Series):
    dtype: DtypeArg = self.dtype or {}
    if is_dict_like(dtype):
        dtype = cast(dict, dtype)
        if col.name in dtype:
            return dtype[col.name]
    col_type = lib.infer_dtype(col, skipna=True)
    from sqlalchemy.types import TIMESTAMP, BigInteger, Boolean, Date, DateTime, Float, Integer, SmallInteger, Text, Time
    if col_type in ('datetime64', 'datetime'):
        try:
            if col.dt.tz is not None:
                return TIMESTAMP(timezone=True)
        except AttributeError:
            if getattr(col, 'tz', None) is not None:
                return TIMESTAMP(timezone=True)
        return DateTime
    if col_type == 'timedelta64':
        warnings.warn("the 'timedelta' type is not supported, and will be written as integer values (ns frequency) to the database.", UserWarning, stacklevel=find_stack_level())
        return BigInteger
    elif col_type == 'floating':
        if col.dtype == 'float32':
            return Float(precision=23)
        else:
            return Float(precision=53)
    elif col_type == 'integer':
        if col.dtype.name.lower() in ('int8', 'uint8', 'int16'):
            return SmallInteger
        elif col.dtype.name.lower() in ('uint16', 'int32'):
            return Integer
        elif col.dtype.name.lower() == 'uint64':
            raise ValueError('Unsigned 64 bit integer datatype is not supported')
        else:
            return BigInteger
    elif col_type == 'boolean':
        return Boolean
    elif col_type == 'date':
        return Date
    elif col_type == 'time':
        return Time
    elif col_type == 'complex':
        raise ValueError('Complex datatypes not supported')
    return Text