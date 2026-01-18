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
def _validate_dt64_dtype(dtype):
    """
    Check that a dtype, if passed, represents either a numpy datetime64[ns]
    dtype or a pandas DatetimeTZDtype.

    Parameters
    ----------
    dtype : object

    Returns
    -------
    dtype : None, numpy.dtype, or DatetimeTZDtype

    Raises
    ------
    ValueError : invalid dtype

    Notes
    -----
    Unlike _validate_tz_from_dtype, this does _not_ allow non-existent
    tz errors to go through
    """
    if dtype is not None:
        dtype = pandas_dtype(dtype)
        if dtype == np.dtype('M8'):
            msg = "Passing in 'datetime64' dtype with no precision is not allowed. Please pass in 'datetime64[ns]' instead."
            raise ValueError(msg)
        if isinstance(dtype, np.dtype) and (dtype.kind != 'M' or not is_supported_dtype(dtype)) or not isinstance(dtype, (np.dtype, DatetimeTZDtype)):
            raise ValueError(f"Unexpected value for 'dtype': '{dtype}'. Must be 'datetime64[s]', 'datetime64[ms]', 'datetime64[us]', 'datetime64[ns]' or DatetimeTZDtype'.")
        if getattr(dtype, 'tz', None):
            dtype = cast(DatetimeTZDtype, dtype)
            dtype = DatetimeTZDtype(unit=dtype.unit, tz=timezones.tz_standardize(dtype.tz))
    return dtype