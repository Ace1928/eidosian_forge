from datetime import timedelta
import pytest
import pytz
from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
from pandas.errors import PerformanceWarning
from pandas import DatetimeIndex
import pandas._testing as tm
from pandas.util.version import Version
def _test_all_offsets(self, n, **kwds):
    valid_offsets = self.valid_date_offsets_plural if n > 1 else self.valid_date_offsets_singular
    for name in valid_offsets:
        self._test_offset(offset_name=name, offset_n=n, **kwds)