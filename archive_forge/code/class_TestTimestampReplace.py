from datetime import datetime
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
import pandas.util._test_decorators as td
import pandas._testing as tm
class TestTimestampReplace:

    def test_replace_out_of_pydatetime_bounds(self):
        ts = Timestamp('2016-01-01').as_unit('ns')
        msg = "Out of bounds timestamp: 99999-01-01 00:00:00 with frequency 'ns'"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            ts.replace(year=99999)
        ts = ts.as_unit('ms')
        result = ts.replace(year=99999)
        assert result.year == 99999
        assert result._value == Timestamp(np.datetime64('99999-01-01', 'ms'))._value

    def test_replace_non_nano(self):
        ts = Timestamp._from_value_and_reso(91514880000000000, NpyDatetimeUnit.NPY_FR_us.value, None)
        assert ts.to_pydatetime() == datetime(4869, 12, 28)
        result = ts.replace(year=4900)
        assert result._creso == ts._creso
        assert result.to_pydatetime() == datetime(4900, 12, 28)

    def test_replace_naive(self):
        ts = Timestamp('2016-01-01 09:00:00')
        result = ts.replace(hour=0)
        expected = Timestamp('2016-01-01 00:00:00')
        assert result == expected

    def test_replace_aware(self, tz_aware_fixture):
        tz = tz_aware_fixture
        ts = Timestamp('2016-01-01 09:00:00', tz=tz)
        result = ts.replace(hour=0)
        expected = Timestamp('2016-01-01 00:00:00', tz=tz)
        assert result == expected

    def test_replace_preserves_nanos(self, tz_aware_fixture):
        tz = tz_aware_fixture
        ts = Timestamp('2016-01-01 09:00:00.000000123', tz=tz)
        result = ts.replace(hour=0)
        expected = Timestamp('2016-01-01 00:00:00.000000123', tz=tz)
        assert result == expected

    def test_replace_multiple(self, tz_aware_fixture):
        tz = tz_aware_fixture
        ts = Timestamp('2016-01-01 09:00:00.000000123', tz=tz)
        result = ts.replace(year=2015, month=2, day=2, hour=0, minute=5, second=5, microsecond=5, nanosecond=5)
        expected = Timestamp('2015-02-02 00:05:05.000005005', tz=tz)
        assert result == expected

    def test_replace_invalid_kwarg(self, tz_aware_fixture):
        tz = tz_aware_fixture
        ts = Timestamp('2016-01-01 09:00:00.000000123', tz=tz)
        msg = 'replace\\(\\) got an unexpected keyword argument'
        with pytest.raises(TypeError, match=msg):
            ts.replace(foo=5)

    def test_replace_integer_args(self, tz_aware_fixture):
        tz = tz_aware_fixture
        ts = Timestamp('2016-01-01 09:00:00.000000123', tz=tz)
        msg = "value must be an integer, received <class 'float'> for hour"
        with pytest.raises(ValueError, match=msg):
            ts.replace(hour=0.1)

    def test_replace_tzinfo_equiv_tz_localize_none(self):
        ts = Timestamp('2013-11-03 01:59:59.999999-0400', tz='US/Eastern')
        assert ts.tz_localize(None) == ts.replace(tzinfo=None)

    @td.skip_if_windows
    def test_replace_tzinfo(self):
        dt = datetime(2016, 3, 27, 1)
        tzinfo = pytz.timezone('CET').localize(dt, is_dst=False).tzinfo
        result_dt = dt.replace(tzinfo=tzinfo)
        result_pd = Timestamp(dt).replace(tzinfo=tzinfo)
        with tm.set_timezone('UTC'):
            assert result_dt.timestamp() == result_pd.timestamp()
        assert result_dt == result_pd
        assert result_dt == result_pd.to_pydatetime()
        result_dt = dt.replace(tzinfo=tzinfo).replace(tzinfo=None)
        result_pd = Timestamp(dt).replace(tzinfo=tzinfo).replace(tzinfo=None)
        with tm.set_timezone('UTC'):
            assert result_dt.timestamp() == result_pd.timestamp()
        assert result_dt == result_pd
        assert result_dt == result_pd.to_pydatetime()

    @pytest.mark.parametrize('tz, normalize', [(pytz.timezone('US/Eastern'), lambda x: x.tzinfo.normalize(x)), (gettz('US/Eastern'), lambda x: x)])
    def test_replace_across_dst(self, tz, normalize):
        ts_naive = Timestamp('2017-12-03 16:03:30')
        ts_aware = conversion.localize_pydatetime(ts_naive, tz)
        assert ts_aware == normalize(ts_aware)
        ts2 = ts_aware.replace(month=6)
        assert (ts2.hour, ts2.minute) == (ts_aware.hour, ts_aware.minute)
        ts2b = normalize(ts2)
        assert ts2 == ts2b

    @pytest.mark.parametrize('unit', ['ns', 'us', 'ms', 's'])
    def test_replace_dst_border(self, unit):
        t = Timestamp('2013-11-3', tz='America/Chicago').as_unit(unit)
        result = t.replace(hour=3)
        expected = Timestamp('2013-11-3 03:00:00', tz='America/Chicago')
        assert result == expected
        assert result._creso == getattr(NpyDatetimeUnit, f'NPY_FR_{unit}').value

    @pytest.mark.parametrize('fold', [0, 1])
    @pytest.mark.parametrize('tz', ['dateutil/Europe/London', 'Europe/London'])
    @pytest.mark.parametrize('unit', ['ns', 'us', 'ms', 's'])
    def test_replace_dst_fold(self, fold, tz, unit):
        d = datetime(2019, 10, 27, 2, 30)
        ts = Timestamp(d, tz=tz).as_unit(unit)
        result = ts.replace(hour=1, fold=fold)
        expected = Timestamp(datetime(2019, 10, 27, 1, 30)).tz_localize(tz, ambiguous=not fold)
        assert result == expected
        assert result._creso == getattr(NpyDatetimeUnit, f'NPY_FR_{unit}').value

    @pytest.mark.parametrize('fold', [0, 1])
    def test_replace_preserves_fold(self, fold):
        tz = gettz('Europe/Moscow')
        ts = Timestamp(year=2009, month=10, day=25, hour=2, minute=30, fold=fold, tzinfo=tz)
        ts_replaced = ts.replace(second=1)
        assert ts_replaced.fold == fold