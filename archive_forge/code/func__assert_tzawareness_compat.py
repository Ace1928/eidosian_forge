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
def _assert_tzawareness_compat(self, other) -> None:
    other_tz = getattr(other, 'tzinfo', None)
    other_dtype = getattr(other, 'dtype', None)
    if isinstance(other_dtype, DatetimeTZDtype):
        other_tz = other.dtype.tz
    if other is NaT:
        pass
    elif self.tz is None:
        if other_tz is not None:
            raise TypeError('Cannot compare tz-naive and tz-aware datetime-like objects.')
    elif other_tz is None:
        raise TypeError('Cannot compare tz-naive and tz-aware datetime-like objects')