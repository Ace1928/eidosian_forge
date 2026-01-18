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
def _validate_tz_from_dtype(dtype, tz: tzinfo | None, explicit_tz_none: bool=False) -> tzinfo | None:
    """
    If the given dtype is a DatetimeTZDtype, extract the implied
    tzinfo object from it and check that it does not conflict with the given
    tz.

    Parameters
    ----------
    dtype : dtype, str
    tz : None, tzinfo
    explicit_tz_none : bool, default False
        Whether tz=None was passed explicitly, as opposed to lib.no_default.

    Returns
    -------
    tz : consensus tzinfo

    Raises
    ------
    ValueError : on tzinfo mismatch
    """
    if dtype is not None:
        if isinstance(dtype, str):
            try:
                dtype = DatetimeTZDtype.construct_from_string(dtype)
            except TypeError:
                pass
        dtz = getattr(dtype, 'tz', None)
        if dtz is not None:
            if tz is not None and (not timezones.tz_compare(tz, dtz)):
                raise ValueError('cannot supply both a tz and a dtype with a tz')
            if explicit_tz_none:
                raise ValueError('Cannot pass both a timezone-aware dtype and tz=None')
            tz = dtz
        if tz is not None and lib.is_np_dtype(dtype, 'M'):
            if tz is not None and (not timezones.tz_compare(tz, dtz)):
                raise ValueError('cannot supply both a tz and a timezone-naive dtype (i.e. datetime64[ns])')
    return tz