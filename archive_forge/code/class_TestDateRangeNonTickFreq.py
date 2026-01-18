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
class TestDateRangeNonTickFreq:

    def test_date_range_custom_business_month_begin(self, unit):
        hcal = USFederalHolidayCalendar()
        freq = offsets.CBMonthBegin(calendar=hcal)
        dti = date_range(start='20120101', end='20130101', freq=freq, unit=unit)
        assert all((freq.is_on_offset(x) for x in dti))
        expected = DatetimeIndex(['2012-01-03', '2012-02-01', '2012-03-01', '2012-04-02', '2012-05-01', '2012-06-01', '2012-07-02', '2012-08-01', '2012-09-04', '2012-10-01', '2012-11-01', '2012-12-03'], dtype=f'M8[{unit}]', freq=freq)
        tm.assert_index_equal(dti, expected)

    def test_date_range_custom_business_month_end(self, unit):
        hcal = USFederalHolidayCalendar()
        freq = offsets.CBMonthEnd(calendar=hcal)
        dti = date_range(start='20120101', end='20130101', freq=freq, unit=unit)
        assert all((freq.is_on_offset(x) for x in dti))
        expected = DatetimeIndex(['2012-01-31', '2012-02-29', '2012-03-30', '2012-04-30', '2012-05-31', '2012-06-29', '2012-07-31', '2012-08-31', '2012-09-28', '2012-10-31', '2012-11-30', '2012-12-31'], dtype=f'M8[{unit}]', freq=freq)
        tm.assert_index_equal(dti, expected)

    def test_date_range_with_custom_holidays(self, unit):
        freq = offsets.CustomBusinessHour(start='15:00', holidays=['2020-11-26'])
        result = date_range(start='2020-11-25 15:00', periods=4, freq=freq, unit=unit)
        expected = DatetimeIndex(['2020-11-25 15:00:00', '2020-11-25 16:00:00', '2020-11-27 15:00:00', '2020-11-27 16:00:00'], dtype=f'M8[{unit}]', freq=freq)
        tm.assert_index_equal(result, expected)

    def test_date_range_businesshour(self, unit):
        idx = DatetimeIndex(['2014-07-04 09:00', '2014-07-04 10:00', '2014-07-04 11:00', '2014-07-04 12:00', '2014-07-04 13:00', '2014-07-04 14:00', '2014-07-04 15:00', '2014-07-04 16:00'], dtype=f'M8[{unit}]', freq='bh')
        rng = date_range('2014-07-04 09:00', '2014-07-04 16:00', freq='bh', unit=unit)
        tm.assert_index_equal(idx, rng)
        idx = DatetimeIndex(['2014-07-04 16:00', '2014-07-07 09:00'], dtype=f'M8[{unit}]', freq='bh')
        rng = date_range('2014-07-04 16:00', '2014-07-07 09:00', freq='bh', unit=unit)
        tm.assert_index_equal(idx, rng)
        idx = DatetimeIndex(['2014-07-04 09:00', '2014-07-04 10:00', '2014-07-04 11:00', '2014-07-04 12:00', '2014-07-04 13:00', '2014-07-04 14:00', '2014-07-04 15:00', '2014-07-04 16:00', '2014-07-07 09:00', '2014-07-07 10:00', '2014-07-07 11:00', '2014-07-07 12:00', '2014-07-07 13:00', '2014-07-07 14:00', '2014-07-07 15:00', '2014-07-07 16:00', '2014-07-08 09:00', '2014-07-08 10:00', '2014-07-08 11:00', '2014-07-08 12:00', '2014-07-08 13:00', '2014-07-08 14:00', '2014-07-08 15:00', '2014-07-08 16:00'], dtype=f'M8[{unit}]', freq='bh')
        rng = date_range('2014-07-04 09:00', '2014-07-08 16:00', freq='bh', unit=unit)
        tm.assert_index_equal(idx, rng)

    def test_date_range_business_hour2(self, unit):
        idx1 = date_range(start='2014-07-04 15:00', end='2014-07-08 10:00', freq='bh', unit=unit)
        idx2 = date_range(start='2014-07-04 15:00', periods=12, freq='bh', unit=unit)
        idx3 = date_range(end='2014-07-08 10:00', periods=12, freq='bh', unit=unit)
        expected = DatetimeIndex(['2014-07-04 15:00', '2014-07-04 16:00', '2014-07-07 09:00', '2014-07-07 10:00', '2014-07-07 11:00', '2014-07-07 12:00', '2014-07-07 13:00', '2014-07-07 14:00', '2014-07-07 15:00', '2014-07-07 16:00', '2014-07-08 09:00', '2014-07-08 10:00'], dtype=f'M8[{unit}]', freq='bh')
        tm.assert_index_equal(idx1, expected)
        tm.assert_index_equal(idx2, expected)
        tm.assert_index_equal(idx3, expected)
        idx4 = date_range(start='2014-07-04 15:45', end='2014-07-08 10:45', freq='bh', unit=unit)
        idx5 = date_range(start='2014-07-04 15:45', periods=12, freq='bh', unit=unit)
        idx6 = date_range(end='2014-07-08 10:45', periods=12, freq='bh', unit=unit)
        expected2 = expected + Timedelta(minutes=45).as_unit(unit)
        expected2.freq = 'bh'
        tm.assert_index_equal(idx4, expected2)
        tm.assert_index_equal(idx5, expected2)
        tm.assert_index_equal(idx6, expected2)

    def test_date_range_business_hour_short(self, unit):
        idx4 = date_range(start='2014-07-01 10:00', freq='bh', periods=1, unit=unit)
        expected4 = DatetimeIndex(['2014-07-01 10:00'], dtype=f'M8[{unit}]', freq='bh')
        tm.assert_index_equal(idx4, expected4)

    def test_date_range_year_start(self, unit):
        rng = date_range('1/1/2013', '7/1/2017', freq='YS', unit=unit)
        exp = DatetimeIndex(['2013-01-01', '2014-01-01', '2015-01-01', '2016-01-01', '2017-01-01'], dtype=f'M8[{unit}]', freq='YS')
        tm.assert_index_equal(rng, exp)

    def test_date_range_year_end(self, unit):
        rng = date_range('1/1/2013', '7/1/2017', freq='YE', unit=unit)
        exp = DatetimeIndex(['2013-12-31', '2014-12-31', '2015-12-31', '2016-12-31'], dtype=f'M8[{unit}]', freq='YE')
        tm.assert_index_equal(rng, exp)

    def test_date_range_negative_freq_year_end(self, unit):
        rng = date_range('2011-12-31', freq='-2YE', periods=3, unit=unit)
        exp = DatetimeIndex(['2011-12-31', '2009-12-31', '2007-12-31'], dtype=f'M8[{unit}]', freq='-2YE')
        tm.assert_index_equal(rng, exp)
        assert rng.freq == '-2YE'

    def test_date_range_business_year_end_year(self, unit):
        rng = date_range('1/1/2013', '7/1/2017', freq='BYE', unit=unit)
        exp = DatetimeIndex(['2013-12-31', '2014-12-31', '2015-12-31', '2016-12-30'], dtype=f'M8[{unit}]', freq='BYE')
        tm.assert_index_equal(rng, exp)

    def test_date_range_bms(self, unit):
        result = date_range('1/1/2000', periods=10, freq='BMS', unit=unit)
        expected = DatetimeIndex(['2000-01-03', '2000-02-01', '2000-03-01', '2000-04-03', '2000-05-01', '2000-06-01', '2000-07-03', '2000-08-01', '2000-09-01', '2000-10-02'], dtype=f'M8[{unit}]', freq='BMS')
        tm.assert_index_equal(result, expected)

    def test_date_range_semi_month_begin(self, unit):
        dates = [datetime(2007, 12, 15), datetime(2008, 1, 1), datetime(2008, 1, 15), datetime(2008, 2, 1), datetime(2008, 2, 15), datetime(2008, 3, 1), datetime(2008, 3, 15), datetime(2008, 4, 1), datetime(2008, 4, 15), datetime(2008, 5, 1), datetime(2008, 5, 15), datetime(2008, 6, 1), datetime(2008, 6, 15), datetime(2008, 7, 1), datetime(2008, 7, 15), datetime(2008, 8, 1), datetime(2008, 8, 15), datetime(2008, 9, 1), datetime(2008, 9, 15), datetime(2008, 10, 1), datetime(2008, 10, 15), datetime(2008, 11, 1), datetime(2008, 11, 15), datetime(2008, 12, 1), datetime(2008, 12, 15)]
        result = date_range(start=dates[0], end=dates[-1], freq='SMS', unit=unit)
        exp = DatetimeIndex(dates, dtype=f'M8[{unit}]', freq='SMS')
        tm.assert_index_equal(result, exp)

    def test_date_range_semi_month_end(self, unit):
        dates = [datetime(2007, 12, 31), datetime(2008, 1, 15), datetime(2008, 1, 31), datetime(2008, 2, 15), datetime(2008, 2, 29), datetime(2008, 3, 15), datetime(2008, 3, 31), datetime(2008, 4, 15), datetime(2008, 4, 30), datetime(2008, 5, 15), datetime(2008, 5, 31), datetime(2008, 6, 15), datetime(2008, 6, 30), datetime(2008, 7, 15), datetime(2008, 7, 31), datetime(2008, 8, 15), datetime(2008, 8, 31), datetime(2008, 9, 15), datetime(2008, 9, 30), datetime(2008, 10, 15), datetime(2008, 10, 31), datetime(2008, 11, 15), datetime(2008, 11, 30), datetime(2008, 12, 15), datetime(2008, 12, 31)]
        result = date_range(start=dates[0], end=dates[-1], freq='SME', unit=unit)
        exp = DatetimeIndex(dates, dtype=f'M8[{unit}]', freq='SME')
        tm.assert_index_equal(result, exp)

    def test_date_range_week_of_month(self, unit):
        result = date_range(start='20110101', periods=1, freq='WOM-1MON', unit=unit)
        expected = DatetimeIndex(['2011-01-03'], dtype=f'M8[{unit}]', freq='WOM-1MON')
        tm.assert_index_equal(result, expected)
        result2 = date_range(start='20110101', periods=2, freq='WOM-1MON', unit=unit)
        expected2 = DatetimeIndex(['2011-01-03', '2011-02-07'], dtype=f'M8[{unit}]', freq='WOM-1MON')
        tm.assert_index_equal(result2, expected2)

    def test_date_range_week_of_month2(self, unit):
        result = date_range('2013-1-1', periods=4, freq='WOM-1SAT', unit=unit)
        expected = DatetimeIndex(['2013-01-05', '2013-02-02', '2013-03-02', '2013-04-06'], dtype=f'M8[{unit}]', freq='WOM-1SAT')
        tm.assert_index_equal(result, expected)

    def test_date_range_negative_freq_month_end(self, unit):
        rng = date_range('2011-01-31', freq='-2ME', periods=3, unit=unit)
        exp = DatetimeIndex(['2011-01-31', '2010-11-30', '2010-09-30'], dtype=f'M8[{unit}]', freq='-2ME')
        tm.assert_index_equal(rng, exp)
        assert rng.freq == '-2ME'

    def test_date_range_fy5253(self, unit):
        freq = offsets.FY5253(startingMonth=1, weekday=3, variation='nearest')
        dti = date_range(start='2013-01-01', periods=2, freq=freq, unit=unit)
        expected = DatetimeIndex(['2013-01-31', '2014-01-30'], dtype=f'M8[{unit}]', freq=freq)
        tm.assert_index_equal(dti, expected)

    @pytest.mark.parametrize('freqstr,offset', [('QS', offsets.QuarterBegin(startingMonth=1)), ('BQE', offsets.BQuarterEnd(startingMonth=12)), ('W-SUN', offsets.Week(weekday=6))])
    def test_date_range_freqstr_matches_offset(self, freqstr, offset):
        sdate = datetime(1999, 12, 25)
        edate = datetime(2000, 1, 1)
        idx1 = date_range(start=sdate, end=edate, freq=freqstr)
        idx2 = date_range(start=sdate, end=edate, freq=offset)
        assert len(idx1) == len(idx2)
        assert idx1.freq == idx2.freq