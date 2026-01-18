from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas._libs import lib
from pandas._libs.algos import unique_deltas
from pandas._libs.tslibs import (
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.dtypes import (
from pandas._libs.tslibs.fields import (
from pandas._libs.tslibs.offsets import (
from pandas._libs.tslibs.parsing import get_rule_month
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.common import is_numeric_dtype
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.algorithms import unique
def _is_business_daily(self) -> bool:
    if self.day_deltas != [1, 3]:
        return False
    first_weekday = self.index[0].weekday()
    shifts = np.diff(self.i8values)
    ppd = periods_per_day(self._creso)
    shifts = np.floor_divide(shifts, ppd)
    weekdays = np.mod(first_weekday + np.cumsum(shifts), 7)
    return bool(np.all((weekdays == 0) & (shifts == 3) | (weekdays > 0) & (weekdays <= 4) & (shifts == 1)))