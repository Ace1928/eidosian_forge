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
class TestCustomDateRange:

    def test_constructor(self):
        bdate_range(START, END, freq=CDay())
        bdate_range(START, periods=20, freq=CDay())
        bdate_range(end=START, periods=20, freq=CDay())
        msg = 'periods must be a number, got C'
        with pytest.raises(TypeError, match=msg):
            date_range('2011-1-1', '2012-1-1', 'C')
        with pytest.raises(TypeError, match=msg):
            bdate_range('2011-1-1', '2012-1-1', 'C')

    def test_misc(self):
        end = datetime(2009, 5, 13)
        dr = bdate_range(end=end, periods=20, freq='C')
        firstDate = end - 19 * CDay()
        assert len(dr) == 20
        assert dr[0] == firstDate
        assert dr[-1] == end

    def test_daterange_bug_456(self):
        rng1 = bdate_range('12/5/2011', '12/5/2011', freq='C')
        rng2 = bdate_range('12/2/2011', '12/5/2011', freq='C')
        assert rng2._data.freq == CDay()
        result = rng1.union(rng2)
        assert isinstance(result, DatetimeIndex)

    def test_cdaterange(self, unit):
        result = bdate_range('2013-05-01', periods=3, freq='C', unit=unit)
        expected = DatetimeIndex(['2013-05-01', '2013-05-02', '2013-05-03'], dtype=f'M8[{unit}]', freq='C')
        tm.assert_index_equal(result, expected)
        assert result.freq == expected.freq

    def test_cdaterange_weekmask(self, unit):
        result = bdate_range('2013-05-01', periods=3, freq='C', weekmask='Sun Mon Tue Wed Thu', unit=unit)
        expected = DatetimeIndex(['2013-05-01', '2013-05-02', '2013-05-05'], dtype=f'M8[{unit}]', freq=result.freq)
        tm.assert_index_equal(result, expected)
        assert result.freq == expected.freq
        msg = 'a custom frequency string is required when holidays or weekmask are passed, got frequency B'
        with pytest.raises(ValueError, match=msg):
            bdate_range('2013-05-01', periods=3, weekmask='Sun Mon Tue Wed Thu')

    def test_cdaterange_holidays(self, unit):
        result = bdate_range('2013-05-01', periods=3, freq='C', holidays=['2013-05-01'], unit=unit)
        expected = DatetimeIndex(['2013-05-02', '2013-05-03', '2013-05-06'], dtype=f'M8[{unit}]', freq=result.freq)
        tm.assert_index_equal(result, expected)
        assert result.freq == expected.freq
        msg = 'a custom frequency string is required when holidays or weekmask are passed, got frequency B'
        with pytest.raises(ValueError, match=msg):
            bdate_range('2013-05-01', periods=3, holidays=['2013-05-01'])

    def test_cdaterange_weekmask_and_holidays(self, unit):
        result = bdate_range('2013-05-01', periods=3, freq='C', weekmask='Sun Mon Tue Wed Thu', holidays=['2013-05-01'], unit=unit)
        expected = DatetimeIndex(['2013-05-02', '2013-05-05', '2013-05-06'], dtype=f'M8[{unit}]', freq=result.freq)
        tm.assert_index_equal(result, expected)
        assert result.freq == expected.freq

    def test_cdaterange_holidays_weekmask_requires_freqstr(self):
        msg = 'a custom frequency string is required when holidays or weekmask are passed, got frequency B'
        with pytest.raises(ValueError, match=msg):
            bdate_range('2013-05-01', periods=3, weekmask='Sun Mon Tue Wed Thu', holidays=['2013-05-01'])

    @pytest.mark.parametrize('freq', [freq for freq in prefix_mapping if freq.startswith('C')])
    def test_all_custom_freq(self, freq):
        bdate_range(START, END, freq=freq, weekmask='Mon Wed Fri', holidays=['2009-03-14'])
        bad_freq = freq + 'FOO'
        msg = f'invalid custom frequency string: {bad_freq}'
        with pytest.raises(ValueError, match=msg):
            bdate_range(START, END, freq=bad_freq)

    @pytest.mark.parametrize('start_end', [('2018-01-01T00:00:01.000Z', '2018-01-03T00:00:01.000Z'), ('2018-01-01T00:00:00.010Z', '2018-01-03T00:00:00.010Z'), ('2001-01-01T00:00:00.010Z', '2001-01-03T00:00:00.010Z')])
    def test_range_with_millisecond_resolution(self, start_end):
        start, end = start_end
        result = date_range(start=start, end=end, periods=2, inclusive='left')
        expected = DatetimeIndex([start], dtype='M8[ns, UTC]')
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('start,period,expected', [('2022-07-23 00:00:00+02:00', 1, ['2022-07-25 00:00:00+02:00']), ('2022-07-22 00:00:00+02:00', 1, ['2022-07-22 00:00:00+02:00']), ('2022-07-22 00:00:00+02:00', 2, ['2022-07-22 00:00:00+02:00', '2022-07-25 00:00:00+02:00'])])
    def test_range_with_timezone_and_custombusinessday(self, start, period, expected):
        result = date_range(start=start, periods=period, freq='C')
        expected = DatetimeIndex(expected).as_unit('ns')
        tm.assert_index_equal(result, expected)