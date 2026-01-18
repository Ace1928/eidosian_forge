from __future__ import annotations
from collections import abc
from datetime import date
from functools import partial
from itertools import islice
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.tslibs import (
from pandas._libs.tslibs.conversion import cast_from_unit_vectorized
from pandas._libs.tslibs.parsing import (
from pandas._libs.tslibs.strptime import array_strptime
from pandas._typing import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.arrays import (
from pandas.core.algorithms import unique
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.datetimes import (
from pandas.core.construction import extract_array
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimes import DatetimeIndex
def _to_datetime_with_unit(arg, unit, name, utc: bool, errors: str) -> Index:
    """
    to_datetime specalized to the case where a 'unit' is passed.
    """
    arg = extract_array(arg, extract_numpy=True)
    if isinstance(arg, IntegerArray):
        arr = arg.astype(f'datetime64[{unit}]')
        tz_parsed = None
    else:
        arg = np.asarray(arg)
        if arg.dtype.kind in 'iu':
            arr = arg.astype(f'datetime64[{unit}]', copy=False)
            try:
                arr = astype_overflowsafe(arr, np.dtype('M8[ns]'), copy=False)
            except OutOfBoundsDatetime:
                if errors == 'raise':
                    raise
                arg = arg.astype(object)
                return _to_datetime_with_unit(arg, unit, name, utc, errors)
            tz_parsed = None
        elif arg.dtype.kind == 'f':
            with np.errstate(over='raise'):
                try:
                    arr = cast_from_unit_vectorized(arg, unit=unit)
                except OutOfBoundsDatetime:
                    if errors != 'raise':
                        return _to_datetime_with_unit(arg.astype(object), unit, name, utc, errors)
                    raise OutOfBoundsDatetime(f"cannot convert input with unit '{unit}'")
            arr = arr.view('M8[ns]')
            tz_parsed = None
        else:
            arg = arg.astype(object, copy=False)
            arr, tz_parsed = tslib.array_with_unit_to_datetime(arg, unit, errors=errors)
    if errors == 'ignore':
        result = Index._with_infer(arr, name=name)
    else:
        result = DatetimeIndex(arr, name=name)
    if not isinstance(result, DatetimeIndex):
        return result
    result = result.tz_localize('UTC').tz_convert(tz_parsed)
    if utc:
        if result.tz is None:
            result = result.tz_localize('utc')
        else:
            result = result.tz_convert('utc')
    return result