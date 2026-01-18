from __future__ import annotations
from datetime import (
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import abbrev_to_npy_unit
from pandas.errors import PerformanceWarning
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_inclusive
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
from pandas.core.arrays import datetimelike as dtl
from pandas.core.arrays._ranges import generate_regular_range
import pandas.core.common as com
from pandas.tseries.frequencies import get_period_alias
from pandas.tseries.offsets import (
def _sequence_to_dt64(data: ArrayLike, *, copy: bool=False, tz: tzinfo | None=None, dayfirst: bool=False, yearfirst: bool=False, ambiguous: TimeAmbiguous='raise', out_unit: str | None=None):
    """
    Parameters
    ----------
    data : np.ndarray or ExtensionArray
        dtl.ensure_arraylike_for_datetimelike has already been called.
    copy : bool, default False
    tz : tzinfo or None, default None
    dayfirst : bool, default False
    yearfirst : bool, default False
    ambiguous : str, bool, or arraylike, default 'raise'
        See pandas._libs.tslibs.tzconversion.tz_localize_to_utc.
    out_unit : str or None, default None
        Desired output resolution.

    Returns
    -------
    result : numpy.ndarray
        The sequence converted to a numpy array with dtype ``datetime64[unit]``.
        Where `unit` is "ns" unless specified otherwise by `out_unit`.
    tz : tzinfo or None
        Either the user-provided tzinfo or one inferred from the data.

    Raises
    ------
    TypeError : PeriodDType data is passed
    """
    data, copy = maybe_convert_dtype(data, copy, tz=tz)
    data_dtype = getattr(data, 'dtype', None)
    if out_unit is None:
        out_unit = 'ns'
    out_dtype = np.dtype(f'M8[{out_unit}]')
    if data_dtype == object or is_string_dtype(data_dtype):
        data = cast(np.ndarray, data)
        copy = False
        if lib.infer_dtype(data, skipna=False) == 'integer':
            data = data.astype(np.int64)
        elif tz is not None and ambiguous == 'raise':
            obj_data = np.asarray(data, dtype=object)
            result = tslib.array_to_datetime_with_tz(obj_data, tz=tz, dayfirst=dayfirst, yearfirst=yearfirst, creso=abbrev_to_npy_unit(out_unit))
            return (result, tz)
        else:
            converted, inferred_tz = objects_to_datetime64(data, dayfirst=dayfirst, yearfirst=yearfirst, allow_object=False, out_unit=out_unit or 'ns')
            copy = False
            if tz and inferred_tz:
                result = converted
            elif inferred_tz:
                tz = inferred_tz
                result = converted
            else:
                result, _ = _construct_from_dt64_naive(converted, tz=tz, copy=copy, ambiguous=ambiguous)
            return (result, tz)
        data_dtype = data.dtype
    if isinstance(data_dtype, DatetimeTZDtype):
        data = cast(DatetimeArray, data)
        tz = _maybe_infer_tz(tz, data.tz)
        result = data._ndarray
    elif lib.is_np_dtype(data_dtype, 'M'):
        if isinstance(data, DatetimeArray):
            data = data._ndarray
        data = cast(np.ndarray, data)
        result, copy = _construct_from_dt64_naive(data, tz=tz, copy=copy, ambiguous=ambiguous)
    else:
        if data.dtype != INT64_DTYPE:
            data = data.astype(np.int64, copy=False)
            copy = False
        data = cast(np.ndarray, data)
        result = data.view(out_dtype)
    if copy:
        result = result.copy()
    assert isinstance(result, np.ndarray), type(result)
    assert result.dtype.kind == 'M'
    assert result.dtype != 'M8'
    assert is_supported_dtype(result.dtype)
    return (result, tz)