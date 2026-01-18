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
def _convert_listlike_datetimes(arg, format: str | None, name: Hashable | None=None, utc: bool=False, unit: str | None=None, errors: DateTimeErrorChoices='raise', dayfirst: bool | None=None, yearfirst: bool | None=None, exact: bool=True):
    """
    Helper function for to_datetime. Performs the conversions of 1D listlike
    of dates

    Parameters
    ----------
    arg : list, tuple, ndarray, Series, Index
        date to be parsed
    name : object
        None or string for the Index name
    utc : bool
        Whether to convert/localize timestamps to UTC.
    unit : str
        None or string of the frequency of the passed data
    errors : str
        error handing behaviors from to_datetime, 'raise', 'coerce', 'ignore'
    dayfirst : bool
        dayfirst parsing behavior from to_datetime
    yearfirst : bool
        yearfirst parsing behavior from to_datetime
    exact : bool, default True
        exact format matching behavior from to_datetime

    Returns
    -------
    Index-like of parsed dates
    """
    if isinstance(arg, (list, tuple)):
        arg = np.array(arg, dtype='O')
    elif isinstance(arg, NumpyExtensionArray):
        arg = np.array(arg)
    arg_dtype = getattr(arg, 'dtype', None)
    tz = 'utc' if utc else None
    if isinstance(arg_dtype, DatetimeTZDtype):
        if not isinstance(arg, (DatetimeArray, DatetimeIndex)):
            return DatetimeIndex(arg, tz=tz, name=name)
        if utc:
            arg = arg.tz_convert(None).tz_localize('utc')
        return arg
    elif isinstance(arg_dtype, ArrowDtype) and arg_dtype.type is Timestamp:
        if utc:
            if isinstance(arg, Index):
                arg_array = cast(ArrowExtensionArray, arg.array)
                if arg_dtype.pyarrow_dtype.tz is not None:
                    arg_array = arg_array._dt_tz_convert('UTC')
                else:
                    arg_array = arg_array._dt_tz_localize('UTC')
                arg = Index(arg_array)
            elif arg_dtype.pyarrow_dtype.tz is not None:
                arg = arg._dt_tz_convert('UTC')
            else:
                arg = arg._dt_tz_localize('UTC')
        return arg
    elif lib.is_np_dtype(arg_dtype, 'M'):
        if not is_supported_dtype(arg_dtype):
            arg = astype_overflowsafe(np.asarray(arg), np.dtype('M8[s]'), is_coerce=errors == 'coerce')
        if not isinstance(arg, (DatetimeArray, DatetimeIndex)):
            return DatetimeIndex(arg, tz=tz, name=name)
        elif utc:
            return arg.tz_localize('utc')
        return arg
    elif unit is not None:
        if format is not None:
            raise ValueError('cannot specify both format and unit')
        return _to_datetime_with_unit(arg, unit, name, utc, errors)
    elif getattr(arg, 'ndim', 1) > 1:
        raise TypeError('arg must be a string, datetime, list, tuple, 1-d array, or Series')
    try:
        arg, _ = maybe_convert_dtype(arg, copy=False, tz=libtimezones.maybe_get_tz(tz))
    except TypeError:
        if errors == 'coerce':
            npvalues = np.array(['NaT'], dtype='datetime64[ns]').repeat(len(arg))
            return DatetimeIndex(npvalues, name=name)
        elif errors == 'ignore':
            idx = Index(arg, name=name)
            return idx
        raise
    arg = ensure_object(arg)
    if format is None:
        format = _guess_datetime_format_for_array(arg, dayfirst=dayfirst)
    if format is not None and format != 'mixed':
        return _array_strptime_with_fallback(arg, name, utc, format, exact, errors)
    result, tz_parsed = objects_to_datetime64(arg, dayfirst=dayfirst, yearfirst=yearfirst, utc=utc, errors=errors, allow_object=True)
    if tz_parsed is not None:
        out_unit = np.datetime_data(result.dtype)[0]
        dtype = cast(DatetimeTZDtype, tz_to_dtype(tz_parsed, out_unit))
        dt64_values = result.view(f'M8[{dtype.unit}]')
        dta = DatetimeArray._simple_new(dt64_values, dtype=dtype)
        return DatetimeIndex._simple_new(dta, name=name)
    return _box_as_indexlike(result, utc=utc, name=name)