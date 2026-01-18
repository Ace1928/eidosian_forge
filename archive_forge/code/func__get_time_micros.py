from __future__ import annotations
import datetime as dt
import operator
from typing import TYPE_CHECKING
import warnings
import numpy as np
import pytz
from pandas._libs import (
from pandas._libs.tslibs import (
from pandas._libs.tslibs.offsets import prefix_mapping
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import is_scalar
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import is_valid_na_for_dtype
from pandas.core.arrays.datetimes import (
import pandas.core.common as com
from pandas.core.indexes.base import (
from pandas.core.indexes.datetimelike import DatetimeTimedeltaMixin
from pandas.core.indexes.extension import inherit_names
from pandas.core.tools.times import to_time
from pandas._libs.tslibs.dtypes import OFFSET_TO_PERIOD_FREQSTR
def _get_time_micros(self) -> npt.NDArray[np.int64]:
    """
        Return the number of microseconds since midnight.

        Returns
        -------
        ndarray[int64_t]
        """
    values = self._data._local_timestamps()
    ppd = periods_per_day(self._data._creso)
    frac = values % ppd
    if self.unit == 'ns':
        micros = frac // 1000
    elif self.unit == 'us':
        micros = frac
    elif self.unit == 'ms':
        micros = frac * 1000
    elif self.unit == 's':
        micros = frac * 1000000
    else:
        raise NotImplementedError(self.unit)
    micros[self._isnan] = -1
    return micros