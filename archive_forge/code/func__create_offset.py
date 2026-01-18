from __future__ import annotations
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import (
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.errors import PerformanceWarning
from pandas import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import WeekDay
from pandas.tseries import offsets
from pandas.tseries.offsets import (
def _create_offset(klass, value=1, normalize=False):
    if klass is FY5253:
        klass = klass(n=value, startingMonth=1, weekday=1, variation='last', normalize=normalize)
    elif klass is FY5253Quarter:
        klass = klass(n=value, startingMonth=1, weekday=1, qtr_with_extra_week=1, variation='last', normalize=normalize)
    elif klass is LastWeekOfMonth:
        klass = klass(n=value, weekday=5, normalize=normalize)
    elif klass is WeekOfMonth:
        klass = klass(n=value, week=1, weekday=5, normalize=normalize)
    elif klass is Week:
        klass = klass(n=value, weekday=5, normalize=normalize)
    elif klass is DateOffset:
        klass = klass(days=value, normalize=normalize)
    else:
        klass = klass(value, normalize=normalize)
    return klass