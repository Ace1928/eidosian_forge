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
class TestOffsetNames:

    def test_get_offset_name(self):
        assert BDay().freqstr == 'B'
        assert BDay(2).freqstr == '2B'
        assert BMonthEnd().freqstr == 'BME'
        assert Week(weekday=0).freqstr == 'W-MON'
        assert Week(weekday=1).freqstr == 'W-TUE'
        assert Week(weekday=2).freqstr == 'W-WED'
        assert Week(weekday=3).freqstr == 'W-THU'
        assert Week(weekday=4).freqstr == 'W-FRI'
        assert LastWeekOfMonth(weekday=WeekDay.SUN).freqstr == 'LWOM-SUN'