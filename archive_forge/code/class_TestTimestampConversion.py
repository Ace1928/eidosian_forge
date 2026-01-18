import calendar
from datetime import (
import locale
import time
import unicodedata
from dateutil.tz import (
from hypothesis import (
import numpy as np
import pytest
import pytz
from pytz import utc
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.timezones import (
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
class TestTimestampConversion:

    def test_conversion(self):
        ts = Timestamp('2000-01-01').as_unit('ns')
        result = ts.to_pydatetime()
        expected = datetime(2000, 1, 1)
        assert result == expected
        assert type(result) == type(expected)
        result = ts.to_datetime64()
        expected = np.datetime64(ts._value, 'ns')
        assert result == expected
        assert type(result) == type(expected)
        assert result.dtype == expected.dtype

    def test_to_period_tz_warning(self):
        ts = Timestamp('2009-04-15 16:17:18', tz='US/Eastern')
        with tm.assert_produces_warning(UserWarning):
            ts.to_period('D')

    def test_to_numpy_alias(self):
        ts = Timestamp(datetime.now())
        assert ts.to_datetime64() == ts.to_numpy()
        msg = 'dtype and copy arguments are ignored'
        with pytest.raises(ValueError, match=msg):
            ts.to_numpy('M8[s]')
        with pytest.raises(ValueError, match=msg):
            ts.to_numpy(copy=True)