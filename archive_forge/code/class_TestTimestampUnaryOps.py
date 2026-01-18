from datetime import datetime
from dateutil.tz import gettz
from hypothesis import (
import numpy as np
import pytest
import pytz
from pytz import utc
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
import pandas.util._test_decorators as td
import pandas._testing as tm
class TestTimestampUnaryOps:

    def test_round_divison_by_zero_raises(self):
        ts = Timestamp('2016-01-01')
        msg = 'Division by zero in rounding'
        with pytest.raises(ValueError, match=msg):
            ts.round('0ns')

    @pytest.mark.parametrize('timestamp, freq, expected', [('20130101 09:10:11', 'D', '20130101'), ('20130101 19:10:11', 'D', '20130102'), ('20130201 12:00:00', 'D', '20130202'), ('20130104 12:00:00', 'D', '20130105'), ('2000-01-05 05:09:15.13', 'D', '2000-01-05 00:00:00'), ('2000-01-05 05:09:15.13', 'H', '2000-01-05 05:00:00'), ('2000-01-05 05:09:15.13', 'S', '2000-01-05 05:09:15')])
    def test_round_frequencies(self, timestamp, freq, expected):
        dt = Timestamp(timestamp)
        result = dt.round(freq)
        expected = Timestamp(expected)
        assert result == expected

    def test_round_tzaware(self):
        dt = Timestamp('20130101 09:10:11', tz='US/Eastern')
        result = dt.round('D')
        expected = Timestamp('20130101', tz='US/Eastern')
        assert result == expected
        dt = Timestamp('20130101 09:10:11', tz='US/Eastern')
        result = dt.round('s')
        assert result == dt

    def test_round_30min(self):
        dt = Timestamp('20130104 12:32:00')
        result = dt.round('30Min')
        expected = Timestamp('20130104 12:30:00')
        assert result == expected

    def test_round_subsecond(self):
        result = Timestamp('2016-10-17 12:00:00.0015').round('ms')
        expected = Timestamp('2016-10-17 12:00:00.002000')
        assert result == expected
        result = Timestamp('2016-10-17 12:00:00.00149').round('ms')
        expected = Timestamp('2016-10-17 12:00:00.001000')
        assert result == expected
        ts = Timestamp('2016-10-17 12:00:00.0015')
        for freq in ['us', 'ns']:
            assert ts == ts.round(freq)
        result = Timestamp('2016-10-17 12:00:00.001501031').round('10ns')
        expected = Timestamp('2016-10-17 12:00:00.001501030')
        assert result == expected

    def test_round_nonstandard_freq(self):
        with tm.assert_produces_warning(False):
            Timestamp('2016-10-17 12:00:00.001501031').round('1010ns')

    def test_round_invalid_arg(self):
        stamp = Timestamp('2000-01-05 05:09:15.13')
        with pytest.raises(ValueError, match=INVALID_FREQ_ERR_MSG):
            stamp.round('foo')

    @pytest.mark.parametrize('test_input, rounder, freq, expected', [('2117-01-01 00:00:45', 'floor', '15s', '2117-01-01 00:00:45'), ('2117-01-01 00:00:45', 'ceil', '15s', '2117-01-01 00:00:45'), ('2117-01-01 00:00:45.000000012', 'floor', '10ns', '2117-01-01 00:00:45.000000010'), ('1823-01-01 00:00:01.000000012', 'ceil', '10ns', '1823-01-01 00:00:01.000000020'), ('1823-01-01 00:00:01', 'floor', '1s', '1823-01-01 00:00:01'), ('1823-01-01 00:00:01', 'ceil', '1s', '1823-01-01 00:00:01'), ('NaT', 'floor', '1s', 'NaT'), ('NaT', 'ceil', '1s', 'NaT')])
    def test_ceil_floor_edge(self, test_input, rounder, freq, expected):
        dt = Timestamp(test_input)
        func = getattr(dt, rounder)
        result = func(freq)
        if dt is NaT:
            assert result is NaT
        else:
            expected = Timestamp(expected)
            assert result == expected

    @pytest.mark.parametrize('test_input, freq, expected', [('2018-01-01 00:02:06', '2s', '2018-01-01 00:02:06'), ('2018-01-01 00:02:00', '2T', '2018-01-01 00:02:00'), ('2018-01-01 00:04:00', '4T', '2018-01-01 00:04:00'), ('2018-01-01 00:15:00', '15T', '2018-01-01 00:15:00'), ('2018-01-01 00:20:00', '20T', '2018-01-01 00:20:00'), ('2018-01-01 03:00:00', '3H', '2018-01-01 03:00:00')])
    @pytest.mark.parametrize('rounder', ['ceil', 'floor', 'round'])
    def test_round_minute_freq(self, test_input, freq, expected, rounder):
        dt = Timestamp(test_input)
        expected = Timestamp(expected)
        func = getattr(dt, rounder)
        result = func(freq)
        assert result == expected

    @pytest.mark.parametrize('unit', ['ns', 'us', 'ms', 's'])
    def test_ceil(self, unit):
        dt = Timestamp('20130101 09:10:11').as_unit(unit)
        result = dt.ceil('D')
        expected = Timestamp('20130102')
        assert result == expected
        assert result._creso == dt._creso

    @pytest.mark.parametrize('unit', ['ns', 'us', 'ms', 's'])
    def test_floor(self, unit):
        dt = Timestamp('20130101 09:10:11').as_unit(unit)
        result = dt.floor('D')
        expected = Timestamp('20130101')
        assert result == expected
        assert result._creso == dt._creso

    @pytest.mark.parametrize('method', ['ceil', 'round', 'floor'])
    @pytest.mark.parametrize('unit', ['ns', 'us', 'ms', 's'])
    def test_round_dst_border_ambiguous(self, method, unit):
        ts = Timestamp('2017-10-29 00:00:00', tz='UTC').tz_convert('Europe/Madrid')
        ts = ts.as_unit(unit)
        result = getattr(ts, method)('H', ambiguous=True)
        assert result == ts
        assert result._creso == getattr(NpyDatetimeUnit, f'NPY_FR_{unit}').value
        result = getattr(ts, method)('H', ambiguous=False)
        expected = Timestamp('2017-10-29 01:00:00', tz='UTC').tz_convert('Europe/Madrid')
        assert result == expected
        assert result._creso == getattr(NpyDatetimeUnit, f'NPY_FR_{unit}').value
        result = getattr(ts, method)('H', ambiguous='NaT')
        assert result is NaT
        msg = 'Cannot infer dst time'
        with pytest.raises(pytz.AmbiguousTimeError, match=msg):
            getattr(ts, method)('H', ambiguous='raise')

    @pytest.mark.parametrize('method, ts_str, freq', [['ceil', '2018-03-11 01:59:00-0600', '5min'], ['round', '2018-03-11 01:59:00-0600', '5min'], ['floor', '2018-03-11 03:01:00-0500', '2H']])
    @pytest.mark.parametrize('unit', ['ns', 'us', 'ms', 's'])
    def test_round_dst_border_nonexistent(self, method, ts_str, freq, unit):
        ts = Timestamp(ts_str, tz='America/Chicago').as_unit(unit)
        result = getattr(ts, method)(freq, nonexistent='shift_forward')
        expected = Timestamp('2018-03-11 03:00:00', tz='America/Chicago')
        assert result == expected
        assert result._creso == getattr(NpyDatetimeUnit, f'NPY_FR_{unit}').value
        result = getattr(ts, method)(freq, nonexistent='NaT')
        assert result is NaT
        msg = '2018-03-11 02:00:00'
        with pytest.raises(pytz.NonExistentTimeError, match=msg):
            getattr(ts, method)(freq, nonexistent='raise')

    @pytest.mark.parametrize('timestamp', ['2018-01-01 0:0:0.124999360', '2018-01-01 0:0:0.125000367', '2018-01-01 0:0:0.125500', '2018-01-01 0:0:0.126500', '2018-01-01 12:00:00', '2019-01-01 12:00:00'])
    @pytest.mark.parametrize('freq', ['2ns', '3ns', '4ns', '5ns', '6ns', '7ns', '250ns', '500ns', '750ns', '1us', '19us', '250us', '500us', '750us', '1s', '2s', '3s', '1D'])
    def test_round_int64(self, timestamp, freq):
        dt = Timestamp(timestamp).as_unit('ns')
        unit = to_offset(freq).nanos
        result = dt.floor(freq)
        assert result._value % unit == 0, f'floor not a {freq} multiple'
        assert 0 <= dt._value - result._value < unit, 'floor error'
        result = dt.ceil(freq)
        assert result._value % unit == 0, f'ceil not a {freq} multiple'
        assert 0 <= result._value - dt._value < unit, 'ceil error'
        result = dt.round(freq)
        assert result._value % unit == 0, f'round not a {freq} multiple'
        assert abs(result._value - dt._value) <= unit // 2, 'round error'
        if unit % 2 == 0 and abs(result._value - dt._value) == unit // 2:
            assert result._value // unit % 2 == 0, 'round half to even error'

    def test_round_implementation_bounds(self):
        result = Timestamp.min.ceil('s')
        expected = Timestamp(1677, 9, 21, 0, 12, 44)
        assert result == expected
        result = Timestamp.max.floor('s')
        expected = Timestamp.max - Timedelta(854775807)
        assert result == expected
        with pytest.raises(OverflowError, match='value too large'):
            Timestamp.min.floor('s')
        msg = '|'.join(['Python int too large to convert to C long', 'int too big to convert'])
        with pytest.raises(OverflowError, match=msg):
            Timestamp.max.ceil('s')

    @pytest.mark.xfail(reason='Failing on builds', strict=False)
    @given(val=st.integers(iNaT + 1, lib.i8max))
    @pytest.mark.parametrize('method', [Timestamp.round, Timestamp.floor, Timestamp.ceil])
    def test_round_sanity(self, val, method):
        val = np.int64(val)
        ts = Timestamp(val)

        def checker(res, ts, nanos):
            if method is Timestamp.round:
                diff = np.abs((res - ts)._value)
                assert diff <= nanos / 2
            elif method is Timestamp.floor:
                assert res <= ts
            elif method is Timestamp.ceil:
                assert res >= ts
        assert method(ts, 'ns') == ts
        res = method(ts, 'us')
        nanos = 1000
        assert np.abs((res - ts)._value) < nanos
        assert res._value % nanos == 0
        checker(res, ts, nanos)
        res = method(ts, 'ms')
        nanos = 1000000
        assert np.abs((res - ts)._value) < nanos
        assert res._value % nanos == 0
        checker(res, ts, nanos)
        res = method(ts, 's')
        nanos = 1000000000
        assert np.abs((res - ts)._value) < nanos
        assert res._value % nanos == 0
        checker(res, ts, nanos)
        res = method(ts, 'min')
        nanos = 60 * 1000000000
        assert np.abs((res - ts)._value) < nanos
        assert res._value % nanos == 0
        checker(res, ts, nanos)
        res = method(ts, 'h')
        nanos = 60 * 60 * 1000000000
        assert np.abs((res - ts)._value) < nanos
        assert res._value % nanos == 0
        checker(res, ts, nanos)
        res = method(ts, 'D')
        nanos = 24 * 60 * 60 * 1000000000
        assert np.abs((res - ts)._value) < nanos
        assert res._value % nanos == 0
        checker(res, ts, nanos)

    def test_replace_out_of_pydatetime_bounds(self):
        ts = Timestamp('2016-01-01').as_unit('ns')
        msg = 'Out of bounds nanosecond timestamp: 99999-01-01 00:00:00'
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

    @pytest.mark.parametrize('arg', ['2013-11-30', '2013-11-30 12:00:00'])
    @pytest.mark.parametrize('unit', ['ns', 'us', 'ms', 's'])
    def test_normalize(self, tz_naive_fixture, arg, unit):
        tz = tz_naive_fixture
        ts = Timestamp(arg, tz=tz).as_unit(unit)
        result = ts.normalize()
        expected = Timestamp('2013-11-30', tz=tz)
        assert result == expected
        assert result._creso == getattr(NpyDatetimeUnit, f'NPY_FR_{unit}').value

    def test_normalize_pre_epoch_dates(self):
        result = Timestamp('1969-01-01 09:00:00').normalize()
        expected = Timestamp('1969-01-01 00:00:00')
        assert result == expected

    @td.skip_if_windows
    def test_timestamp(self, fixed_now_ts):
        ts = fixed_now_ts
        uts = ts.replace(tzinfo=utc)
        assert ts.timestamp() == uts.timestamp()
        tsc = Timestamp('2014-10-11 11:00:01.12345678', tz='US/Central')
        utsc = tsc.tz_convert('UTC')
        assert tsc.timestamp() == utsc.timestamp()
        with tm.set_timezone('UTC'):
            dt = ts.to_pydatetime()
            assert dt.timestamp() == ts.timestamp()