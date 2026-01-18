from datetime import (
import re
import numpy as np
import pytest
import pytz
from pytz import timezone
from pandas._libs.tslibs import timezones
from pandas._libs.tslibs.offsets import (
from pandas.errors import OutOfBoundsDatetime
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.datetimes import _generate_range as generate_range
from pandas.tests.indexes.datetimes.test_timezones import (
from pandas.tseries.holiday import USFederalHolidayCalendar
class TestDateRangeNonNano:

    def test_date_range_reso_validation(self):
        msg = "'unit' must be one of 's', 'ms', 'us', 'ns'"
        with pytest.raises(ValueError, match=msg):
            date_range('2016-01-01', '2016-03-04', periods=3, unit='h')

    def test_date_range_freq_higher_than_reso(self):
        msg = 'Use a lower freq or a higher unit instead'
        with pytest.raises(ValueError, match=msg):
            date_range('2016-01-01', '2016-01-02', freq='ns', unit='ms')

    def test_date_range_freq_matches_reso(self):
        dti = date_range('2016-01-01', '2016-01-01 00:00:01', freq='ms', unit='ms')
        rng = np.arange(1451606400000, 1451606401001, dtype=np.int64)
        expected = DatetimeIndex(rng.view('M8[ms]'), freq='ms')
        tm.assert_index_equal(dti, expected)
        dti = date_range('2016-01-01', '2016-01-01 00:00:01', freq='us', unit='us')
        rng = np.arange(1451606400000000, 1451606401000001, dtype=np.int64)
        expected = DatetimeIndex(rng.view('M8[us]'), freq='us')
        tm.assert_index_equal(dti, expected)
        dti = date_range('2016-01-01', '2016-01-01 00:00:00.001', freq='ns', unit='ns')
        rng = np.arange(1451606400000000000, 1451606400001000001, dtype=np.int64)
        expected = DatetimeIndex(rng.view('M8[ns]'), freq='ns')
        tm.assert_index_equal(dti, expected)

    def test_date_range_freq_lower_than_endpoints(self):
        start = Timestamp('2022-10-19 11:50:44.719781')
        end = Timestamp('2022-10-19 11:50:47.066458')
        with pytest.raises(ValueError, match='Cannot losslessly convert units'):
            date_range(start, end, periods=3, unit='s')
        dti = date_range(start, end, periods=2, unit='us')
        rng = np.array([start.as_unit('us')._value, end.as_unit('us')._value], dtype=np.int64)
        expected = DatetimeIndex(rng.view('M8[us]'))
        tm.assert_index_equal(dti, expected)

    def test_date_range_non_nano(self):
        start = np.datetime64('1066-10-14')
        end = np.datetime64('2305-07-13')
        dti = date_range(start, end, freq='D', unit='s')
        assert dti.freq == 'D'
        assert dti.dtype == 'M8[s]'
        exp = np.arange(start.astype('M8[s]').view('i8'), (end + 1).astype('M8[s]').view('i8'), 24 * 3600).view('M8[s]')
        tm.assert_numpy_array_equal(dti.to_numpy(), exp)