from datetime import timedelta
import pytest
import pytz
from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
from pandas.errors import PerformanceWarning
from pandas import DatetimeIndex
import pandas._testing as tm
from pandas.util.version import Version
def _make_timestamp(self, string, hrs_offset, tz):
    if hrs_offset >= 0:
        offset_string = f'{hrs_offset:02d}00'
    else:
        offset_string = f'-{hrs_offset * -1:02}00'
    return Timestamp(string + offset_string).tz_convert(tz)