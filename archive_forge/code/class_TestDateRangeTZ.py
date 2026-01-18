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
class TestDateRangeTZ:
    """Tests for date_range with timezones"""

    def test_hongkong_tz_convert(self):
        dr = date_range('2012-01-01', '2012-01-10', freq='D', tz='Hongkong')
        dr.hour

    @pytest.mark.parametrize('tzstr', ['US/Eastern', 'dateutil/US/Eastern'])
    def test_date_range_span_dst_transition(self, tzstr):
        dr = date_range('03/06/2012 00:00', periods=200, freq='W-FRI', tz='US/Eastern')
        assert (dr.hour == 0).all()
        dr = date_range('2012-11-02', periods=10, tz=tzstr)
        result = dr.hour
        expected = pd.Index([0] * 10, dtype='int32')
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('tzstr', ['US/Eastern', 'dateutil/US/Eastern'])
    def test_date_range_timezone_str_argument(self, tzstr):
        tz = timezones.maybe_get_tz(tzstr)
        result = date_range('1/1/2000', periods=10, tz=tzstr)
        expected = date_range('1/1/2000', periods=10, tz=tz)
        tm.assert_index_equal(result, expected)

    def test_date_range_with_fixed_tz(self):
        off = FixedOffset(420, '+07:00')
        start = datetime(2012, 3, 11, 5, 0, 0, tzinfo=off)
        end = datetime(2012, 6, 11, 5, 0, 0, tzinfo=off)
        rng = date_range(start=start, end=end)
        assert off == rng.tz
        rng2 = date_range(start, periods=len(rng), tz=off)
        tm.assert_index_equal(rng, rng2)
        rng3 = date_range('3/11/2012 05:00:00+07:00', '6/11/2012 05:00:00+07:00')
        assert (rng.values == rng3.values).all()

    def test_date_range_with_fixedoffset_noname(self):
        off = fixed_off_no_name
        start = datetime(2012, 3, 11, 5, 0, 0, tzinfo=off)
        end = datetime(2012, 6, 11, 5, 0, 0, tzinfo=off)
        rng = date_range(start=start, end=end)
        assert off == rng.tz
        idx = pd.Index([start, end])
        assert off == idx.tz

    @pytest.mark.parametrize('tzstr', ['US/Eastern', 'dateutil/US/Eastern'])
    def test_date_range_with_tz(self, tzstr):
        stamp = Timestamp('3/11/2012 05:00', tz=tzstr)
        assert stamp.hour == 5
        rng = date_range('3/11/2012 04:00', periods=10, freq='h', tz=tzstr)
        assert stamp == rng[1]

    @pytest.mark.parametrize('tz', ['Europe/London', 'dateutil/Europe/London'])
    def test_date_range_ambiguous_endpoint(self, tz):
        with pytest.raises(pytz.AmbiguousTimeError, match='Cannot infer dst time'):
            date_range('2013-10-26 23:00', '2013-10-27 01:00', tz='Europe/London', freq='h')
        times = date_range('2013-10-26 23:00', '2013-10-27 01:00', freq='h', tz=tz, ambiguous='infer')
        assert times[0] == Timestamp('2013-10-26 23:00', tz=tz)
        assert times[-1] == Timestamp('2013-10-27 01:00:00+0000', tz=tz)

    @pytest.mark.parametrize('tz, option, expected', [['US/Pacific', 'shift_forward', '2019-03-10 03:00'], ['dateutil/US/Pacific', 'shift_forward', '2019-03-10 03:00'], ['US/Pacific', 'shift_backward', '2019-03-10 01:00'], ['dateutil/US/Pacific', 'shift_backward', '2019-03-10 01:00'], ['US/Pacific', timedelta(hours=1), '2019-03-10 03:00']])
    def test_date_range_nonexistent_endpoint(self, tz, option, expected):
        with pytest.raises(pytz.NonExistentTimeError, match='2019-03-10 02:00:00'):
            date_range('2019-03-10 00:00', '2019-03-10 02:00', tz='US/Pacific', freq='h')
        times = date_range('2019-03-10 00:00', '2019-03-10 02:00', freq='h', tz=tz, nonexistent=option)
        assert times[-1] == Timestamp(expected, tz=tz)