import calendar
from collections import deque
from datetime import (
from decimal import Decimal
import locale
from dateutil.parser import parse
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
import pytz
from pandas._libs import tslib
from pandas._libs.tslibs import (
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_datetime64_ns_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
from pandas.core.tools import datetimes as tools
from pandas.core.tools.datetimes import start_caching_at
class TestToDatetime:

    @pytest.mark.filterwarnings('ignore:Could not infer format')
    def test_to_datetime_overflow(self):
        arg = '08335394550'
        msg = 'Parsing "08335394550" to datetime overflows, at position 0'
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(arg)
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime([arg])
        res = to_datetime(arg, errors='coerce')
        assert res is NaT
        res = to_datetime([arg], errors='coerce')
        tm.assert_index_equal(res, Index([NaT]))
        res = to_datetime(arg, errors='ignore')
        assert isinstance(res, str) and res == arg
        res = to_datetime([arg], errors='ignore')
        tm.assert_index_equal(res, Index([arg], dtype=object))

    def test_to_datetime_mixed_datetime_and_string(self):
        d1 = datetime(2020, 1, 1, 17, tzinfo=timezone(-timedelta(hours=1)))
        d2 = datetime(2020, 1, 1, 18, tzinfo=timezone(-timedelta(hours=1)))
        res = to_datetime(['2020-01-01 17:00 -0100', d2])
        expected = to_datetime([d1, d2]).tz_convert(timezone(timedelta(minutes=-60)))
        tm.assert_index_equal(res, expected)

    def test_to_datetime_mixed_string_and_numeric(self):
        vals = ['2016-01-01', 0]
        expected = DatetimeIndex([Timestamp(x) for x in vals])
        result = to_datetime(vals, format='mixed')
        result2 = to_datetime(vals[::-1], format='mixed')[::-1]
        result3 = DatetimeIndex(vals)
        result4 = DatetimeIndex(vals[::-1])[::-1]
        tm.assert_index_equal(result, expected)
        tm.assert_index_equal(result2, expected)
        tm.assert_index_equal(result3, expected)
        tm.assert_index_equal(result4, expected)

    @pytest.mark.parametrize('format', ['%Y-%m-%d', '%Y-%d-%m'], ids=['ISO8601', 'non-ISO8601'])
    def test_to_datetime_mixed_date_and_string(self, format):
        d1 = date(2020, 1, 2)
        res = to_datetime(['2020-01-01', d1], format=format)
        expected = DatetimeIndex(['2020-01-01', '2020-01-02'], dtype='M8[ns]')
        tm.assert_index_equal(res, expected)

    @pytest.mark.parametrize('fmt', ['%Y-%d-%m %H:%M:%S%z', '%Y-%m-%d %H:%M:%S%z'], ids=['non-ISO8601 format', 'ISO8601 format'])
    @pytest.mark.parametrize('utc, args, expected', [pytest.param(True, ['2000-01-01 01:00:00-08:00', '2000-01-01 02:00:00-08:00'], DatetimeIndex(['2000-01-01 09:00:00+00:00', '2000-01-01 10:00:00+00:00'], dtype='datetime64[ns, UTC]'), id='all tz-aware, with utc'), pytest.param(False, ['2000-01-01 01:00:00+00:00', '2000-01-01 02:00:00+00:00'], DatetimeIndex(['2000-01-01 01:00:00+00:00', '2000-01-01 02:00:00+00:00']), id='all tz-aware, without utc'), pytest.param(True, ['2000-01-01 01:00:00-08:00', '2000-01-01 02:00:00+00:00'], DatetimeIndex(['2000-01-01 09:00:00+00:00', '2000-01-01 02:00:00+00:00'], dtype='datetime64[ns, UTC]'), id='all tz-aware, mixed offsets, with utc'), pytest.param(True, ['2000-01-01 01:00:00', '2000-01-01 02:00:00+00:00'], DatetimeIndex(['2000-01-01 01:00:00+00:00', '2000-01-01 02:00:00+00:00'], dtype='datetime64[ns, UTC]'), id='tz-aware string, naive pydatetime, with utc')])
    @pytest.mark.parametrize('constructor', [Timestamp, lambda x: Timestamp(x).to_pydatetime()])
    def test_to_datetime_mixed_datetime_and_string_with_format(self, fmt, utc, args, expected, constructor):
        ts1 = constructor(args[0])
        ts2 = args[1]
        result = to_datetime([ts1, ts2], format=fmt, utc=utc)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('fmt', ['%Y-%d-%m %H:%M:%S%z', '%Y-%m-%d %H:%M:%S%z'], ids=['non-ISO8601 format', 'ISO8601 format'])
    @pytest.mark.parametrize('constructor', [Timestamp, lambda x: Timestamp(x).to_pydatetime()])
    def test_to_datetime_mixed_datetime_and_string_with_format_mixed_offsets_utc_false(self, fmt, constructor):
        args = ['2000-01-01 01:00:00', '2000-01-01 02:00:00+00:00']
        ts1 = constructor(args[0])
        ts2 = args[1]
        msg = 'parsing datetimes with mixed time zones will raise an error'
        expected = Index([Timestamp('2000-01-01 01:00:00'), Timestamp('2000-01-01 02:00:00+0000', tz='UTC')])
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = to_datetime([ts1, ts2], format=fmt, utc=False)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('fmt, expected', [pytest.param('%Y-%m-%d %H:%M:%S%z', Index([Timestamp('2000-01-01 09:00:00+0100', tz='UTC+01:00'), Timestamp('2000-01-02 02:00:00+0200', tz='UTC+02:00'), NaT]), id='ISO8601, non-UTC'), pytest.param('%Y-%d-%m %H:%M:%S%z', Index([Timestamp('2000-01-01 09:00:00+0100', tz='UTC+01:00'), Timestamp('2000-02-01 02:00:00+0200', tz='UTC+02:00'), NaT]), id='non-ISO8601, non-UTC')])
    def test_to_datetime_mixed_offsets_with_none_tz(self, fmt, expected):
        msg = 'parsing datetimes with mixed time zones will raise an error'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = to_datetime(['2000-01-01 09:00:00+01:00', '2000-01-02 02:00:00+02:00', None], format=fmt, utc=False)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('fmt, expected', [pytest.param('%Y-%m-%d %H:%M:%S%z', DatetimeIndex(['2000-01-01 08:00:00+00:00', '2000-01-02 00:00:00+00:00', 'NaT'], dtype='datetime64[ns, UTC]'), id='ISO8601, UTC'), pytest.param('%Y-%d-%m %H:%M:%S%z', DatetimeIndex(['2000-01-01 08:00:00+00:00', '2000-02-01 00:00:00+00:00', 'NaT'], dtype='datetime64[ns, UTC]'), id='non-ISO8601, UTC')])
    def test_to_datetime_mixed_offsets_with_none(self, fmt, expected):
        result = to_datetime(['2000-01-01 09:00:00+01:00', '2000-01-02 02:00:00+02:00', None], format=fmt, utc=True)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('fmt', ['%Y-%d-%m %H:%M:%S%z', '%Y-%m-%d %H:%M:%S%z'], ids=['non-ISO8601 format', 'ISO8601 format'])
    @pytest.mark.parametrize('args', [pytest.param(['2000-01-01 01:00:00-08:00', '2000-01-01 02:00:00-07:00'], id='all tz-aware, mixed timezones, without utc')])
    @pytest.mark.parametrize('constructor', [Timestamp, lambda x: Timestamp(x).to_pydatetime()])
    def test_to_datetime_mixed_datetime_and_string_with_format_raises(self, fmt, args, constructor):
        ts1 = constructor(args[0])
        ts2 = constructor(args[1])
        with pytest.raises(ValueError, match='cannot be converted to datetime64 unless utc=True'):
            to_datetime([ts1, ts2], format=fmt, utc=False)

    def test_to_datetime_np_str(self):
        value = np.str_('2019-02-04 10:18:46.297000+0000')
        ser = Series([value])
        exp = Timestamp('2019-02-04 10:18:46.297000', tz='UTC')
        assert to_datetime(value) == exp
        assert to_datetime(ser.iloc[0]) == exp
        res = to_datetime([value])
        expected = Index([exp])
        tm.assert_index_equal(res, expected)
        res = to_datetime(ser)
        expected = Series(expected)
        tm.assert_series_equal(res, expected)

    @pytest.mark.parametrize('s, _format, dt', [['2015-1-1', '%G-%V-%u', datetime(2014, 12, 29, 0, 0)], ['2015-1-4', '%G-%V-%u', datetime(2015, 1, 1, 0, 0)], ['2015-1-7', '%G-%V-%u', datetime(2015, 1, 4, 0, 0)]])
    def test_to_datetime_iso_week_year_format(self, s, _format, dt):
        assert to_datetime(s, format=_format) == dt

    @pytest.mark.parametrize('msg, s, _format', [["ISO week directive '%V' is incompatible with the year directive '%Y'. Use the ISO year '%G' instead.", '1999 50', '%Y %V'], ["ISO year directive '%G' must be used with the ISO week directive '%V' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 51', '%G %V'], ["ISO year directive '%G' must be used with the ISO week directive '%V' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 Monday', '%G %A'], ["ISO year directive '%G' must be used with the ISO week directive '%V' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 Mon', '%G %a'], ["ISO year directive '%G' must be used with the ISO week directive '%V' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 6', '%G %w'], ["ISO year directive '%G' must be used with the ISO week directive '%V' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 6', '%G %u'], ["ISO year directive '%G' must be used with the ISO week directive '%V' and a weekday directive '%A', '%a', '%w', or '%u'.", '2051', '%G'], ["Day of the year directive '%j' is not compatible with ISO year directive '%G'. Use '%Y' instead.", '1999 51 6 256', '%G %V %u %j'], ["ISO week directive '%V' is incompatible with the year directive '%Y'. Use the ISO year '%G' instead.", '1999 51 Sunday', '%Y %V %A'], ["ISO week directive '%V' is incompatible with the year directive '%Y'. Use the ISO year '%G' instead.", '1999 51 Sun', '%Y %V %a'], ["ISO week directive '%V' is incompatible with the year directive '%Y'. Use the ISO year '%G' instead.", '1999 51 1', '%Y %V %w'], ["ISO week directive '%V' is incompatible with the year directive '%Y'. Use the ISO year '%G' instead.", '1999 51 1', '%Y %V %u'], ["ISO week directive '%V' must be used with the ISO year directive '%G' and a weekday directive '%A', '%a', '%w', or '%u'.", '20', '%V'], ["ISO week directive '%V' must be used with the ISO year directive '%G' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 51 Sunday', '%V %A'], ["ISO week directive '%V' must be used with the ISO year directive '%G' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 51 Sun', '%V %a'], ["ISO week directive '%V' must be used with the ISO year directive '%G' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 51 1', '%V %w'], ["ISO week directive '%V' must be used with the ISO year directive '%G' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 51 1', '%V %u'], ["Day of the year directive '%j' is not compatible with ISO year directive '%G'. Use '%Y' instead.", '1999 50', '%G %j'], ["ISO week directive '%V' must be used with the ISO year directive '%G' and a weekday directive '%A', '%a', '%w', or '%u'.", '20 Monday', '%V %A']])
    @pytest.mark.parametrize('errors', ['raise', 'coerce', 'ignore'])
    def test_error_iso_week_year(self, msg, s, _format, errors):
        if locale.getlocale() != ('zh_CN', 'UTF-8') and locale.getlocale() != ('it_IT', 'UTF-8'):
            with pytest.raises(ValueError, match=msg):
                to_datetime(s, format=_format, errors=errors)

    @pytest.mark.parametrize('tz', [None, 'US/Central'])
    def test_to_datetime_dtarr(self, tz):
        dti = date_range('1965-04-03', periods=19, freq='2W', tz=tz)
        arr = dti._data
        result = to_datetime(arr)
        assert result is arr

    @td.skip_if_windows
    @pytest.mark.parametrize('arg_class', [Series, Index])
    @pytest.mark.parametrize('utc', [True, False])
    @pytest.mark.parametrize('tz', [None, 'US/Central'])
    def test_to_datetime_arrow(self, tz, utc, arg_class):
        pa = pytest.importorskip('pyarrow')
        dti = date_range('1965-04-03', periods=19, freq='2W', tz=tz)
        dti = arg_class(dti)
        dti_arrow = dti.astype(pd.ArrowDtype(pa.timestamp(unit='ns', tz=tz)))
        result = to_datetime(dti_arrow, utc=utc)
        expected = to_datetime(dti, utc=utc).astype(pd.ArrowDtype(pa.timestamp(unit='ns', tz=tz if not utc else 'UTC')))
        if not utc and arg_class is not Series:
            assert result is dti_arrow
        if arg_class is Series:
            tm.assert_series_equal(result, expected)
        else:
            tm.assert_index_equal(result, expected)

    def test_to_datetime_pydatetime(self):
        actual = to_datetime(datetime(2008, 1, 15))
        assert actual == datetime(2008, 1, 15)

    def test_to_datetime_YYYYMMDD(self):
        actual = to_datetime('20080115')
        assert actual == datetime(2008, 1, 15)

    def test_to_datetime_unparsable_ignore(self):
        ser = 'Month 1, 1999'
        assert to_datetime(ser, errors='ignore') == ser

    @td.skip_if_windows
    def test_to_datetime_now(self):
        with tm.set_timezone('US/Eastern'):
            now = Timestamp('now').as_unit('ns')
            pdnow = to_datetime('now')
            pdnow2 = to_datetime(['now'])[0]
            assert abs(pdnow._value - now._value) < 10000000000.0
            assert abs(pdnow2._value - now._value) < 10000000000.0
            assert pdnow.tzinfo is None
            assert pdnow2.tzinfo is None

    @td.skip_if_windows
    @pytest.mark.parametrize('tz', ['Pacific/Auckland', 'US/Samoa'])
    def test_to_datetime_today(self, tz):
        with tm.set_timezone(tz):
            nptoday = np.datetime64('today').astype('datetime64[ns]').astype(np.int64)
            pdtoday = to_datetime('today')
            pdtoday2 = to_datetime(['today'])[0]
            tstoday = Timestamp('today').as_unit('ns')
            tstoday2 = Timestamp.today().as_unit('ns')
            assert abs(pdtoday.normalize()._value - nptoday) < 10000000000.0
            assert abs(pdtoday2.normalize()._value - nptoday) < 10000000000.0
            assert abs(pdtoday._value - tstoday._value) < 10000000000.0
            assert abs(pdtoday._value - tstoday2._value) < 10000000000.0
            assert pdtoday.tzinfo is None
            assert pdtoday2.tzinfo is None

    @pytest.mark.parametrize('arg', ['now', 'today'])
    def test_to_datetime_today_now_unicode_bytes(self, arg):
        to_datetime([arg])

    @pytest.mark.parametrize('format, expected_ds', [('%Y-%m-%d %H:%M:%S%z', '2020-01-03'), ('%Y-%d-%m %H:%M:%S%z', '2020-03-01'), (None, '2020-01-03')])
    @pytest.mark.parametrize('string, attribute', [('now', 'utcnow'), ('today', 'today')])
    def test_to_datetime_now_with_format(self, format, expected_ds, string, attribute):
        result = to_datetime(['2020-01-03 00:00:00Z', string], format=format, utc=True)
        expected = DatetimeIndex([expected_ds, getattr(Timestamp, attribute)()], dtype='datetime64[ns, UTC]')
        assert (expected - result).max().total_seconds() < 1

    @pytest.mark.parametrize('dt', [np.datetime64('2000-01-01'), np.datetime64('2000-01-02')])
    def test_to_datetime_dt64s(self, cache, dt):
        assert to_datetime(dt, cache=cache) == Timestamp(dt)

    @pytest.mark.parametrize('arg, format', [('2001-01-01', '%Y-%m-%d'), ('01-01-2001', '%d-%m-%Y')])
    def test_to_datetime_dt64s_and_str(self, arg, format):
        result = to_datetime([arg, np.datetime64('2020-01-01')], format=format)
        expected = DatetimeIndex(['2001-01-01', '2020-01-01'])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('dt', [np.datetime64('1000-01-01'), np.datetime64('5000-01-02')])
    @pytest.mark.parametrize('errors', ['raise', 'ignore', 'coerce'])
    def test_to_datetime_dt64s_out_of_ns_bounds(self, cache, dt, errors):
        ts = to_datetime(dt, errors=errors, cache=cache)
        assert isinstance(ts, Timestamp)
        assert ts.unit == 's'
        assert ts.asm8 == dt
        ts = Timestamp(dt)
        assert ts.unit == 's'
        assert ts.asm8 == dt

    @pytest.mark.skip_ubsan
    def test_to_datetime_dt64d_out_of_bounds(self, cache):
        dt64 = np.datetime64(np.iinfo(np.int64).max, 'D')
        msg = 'Out of bounds second timestamp: 25252734927768524-07-27'
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp(dt64)
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(dt64, errors='raise', cache=cache)
        assert to_datetime(dt64, errors='coerce', cache=cache) is NaT

    @pytest.mark.parametrize('unit', ['s', 'D'])
    def test_to_datetime_array_of_dt64s(self, cache, unit):
        dts = [np.datetime64('2000-01-01', unit), np.datetime64('2000-01-02', unit)] * 30
        result = to_datetime(dts, cache=cache)
        if cache:
            expected = DatetimeIndex([Timestamp(x).asm8 for x in dts], dtype='M8[s]')
        else:
            expected = DatetimeIndex([Timestamp(x).asm8 for x in dts], dtype='M8[ns]')
        tm.assert_index_equal(result, expected)
        dts_with_oob = dts + [np.datetime64('9999-01-01')]
        to_datetime(dts_with_oob, errors='raise')
        result = to_datetime(dts_with_oob, errors='coerce', cache=cache)
        if not cache:
            expected = DatetimeIndex([Timestamp(dts_with_oob[0]).asm8, Timestamp(dts_with_oob[1]).asm8] * 30 + [NaT])
        else:
            expected = DatetimeIndex(np.array(dts_with_oob, dtype='M8[s]'))
        tm.assert_index_equal(result, expected)
        result = to_datetime(dts_with_oob, errors='ignore', cache=cache)
        if not cache:
            expected = Index(dts_with_oob)
        tm.assert_index_equal(result, expected)

    def test_out_of_bounds_errors_ignore(self):
        result = to_datetime(np.datetime64('9999-01-01'), errors='ignore')
        expected = np.datetime64('9999-01-01')
        assert result == expected

    def test_out_of_bounds_errors_ignore2(self):
        msg = "errors='ignore' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            res = to_datetime(Series(['2362-01-01', np.nan], dtype=object), errors='ignore')
        exp = Series(['2362-01-01', np.nan], dtype=object)
        tm.assert_series_equal(res, exp)

    def test_to_datetime_tz(self, cache):
        arr = [Timestamp('2013-01-01 13:00:00-0800', tz='US/Pacific'), Timestamp('2013-01-02 14:00:00-0800', tz='US/Pacific')]
        result = to_datetime(arr, cache=cache)
        expected = DatetimeIndex(['2013-01-01 13:00:00', '2013-01-02 14:00:00'], tz='US/Pacific')
        tm.assert_index_equal(result, expected)

    def test_to_datetime_tz_mixed(self, cache):
        arr = [Timestamp('2013-01-01 13:00:00', tz='US/Pacific'), Timestamp('2013-01-02 14:00:00', tz='US/Eastern')]
        msg = 'Tz-aware datetime.datetime cannot be converted to datetime64 unless utc=True'
        with pytest.raises(ValueError, match=msg):
            to_datetime(arr, cache=cache)
        depr_msg = "errors='ignore' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            result = to_datetime(arr, cache=cache, errors='ignore')
        expected = Index([Timestamp('2013-01-01 13:00:00-08:00'), Timestamp('2013-01-02 14:00:00-05:00')], dtype='object')
        tm.assert_index_equal(result, expected)
        result = to_datetime(arr, cache=cache, errors='coerce')
        expected = DatetimeIndex(['2013-01-01 13:00:00-08:00', 'NaT'], dtype='datetime64[ns, US/Pacific]')
        tm.assert_index_equal(result, expected)

    def test_to_datetime_different_offsets(self, cache):
        ts_string_1 = 'March 1, 2018 12:00:00+0400'
        ts_string_2 = 'March 1, 2018 12:00:00+0500'
        arr = [ts_string_1] * 5 + [ts_string_2] * 5
        expected = Index([parse(x) for x in arr])
        msg = 'parsing datetimes with mixed time zones will raise an error'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = to_datetime(arr, cache=cache)
        tm.assert_index_equal(result, expected)

    def test_to_datetime_tz_pytz(self, cache):
        us_eastern = pytz.timezone('US/Eastern')
        arr = np.array([us_eastern.localize(datetime(year=2000, month=1, day=1, hour=3, minute=0)), us_eastern.localize(datetime(year=2000, month=6, day=1, hour=3, minute=0))], dtype=object)
        result = to_datetime(arr, utc=True, cache=cache)
        expected = DatetimeIndex(['2000-01-01 08:00:00+00:00', '2000-06-01 07:00:00+00:00'], dtype='datetime64[ns, UTC]', freq=None)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('init_constructor, end_constructor', [(Index, DatetimeIndex), (list, DatetimeIndex), (np.array, DatetimeIndex), (Series, Series)])
    def test_to_datetime_utc_true(self, cache, init_constructor, end_constructor):
        data = ['20100102 121314', '20100102 121315']
        expected_data = [Timestamp('2010-01-02 12:13:14', tz='utc'), Timestamp('2010-01-02 12:13:15', tz='utc')]
        result = to_datetime(init_constructor(data), format='%Y%m%d %H%M%S', utc=True, cache=cache)
        expected = end_constructor(expected_data)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('scalar, expected', [['20100102 121314', Timestamp('2010-01-02 12:13:14', tz='utc')], ['20100102 121315', Timestamp('2010-01-02 12:13:15', tz='utc')]])
    def test_to_datetime_utc_true_scalar(self, cache, scalar, expected):
        result = to_datetime(scalar, format='%Y%m%d %H%M%S', utc=True, cache=cache)
        assert result == expected

    def test_to_datetime_utc_true_with_series_single_value(self, cache):
        ts = 1.5e+18
        result = to_datetime(Series([ts]), utc=True, cache=cache)
        expected = Series([Timestamp(ts, tz='utc')])
        tm.assert_series_equal(result, expected)

    def test_to_datetime_utc_true_with_series_tzaware_string(self, cache):
        ts = '2013-01-01 00:00:00-01:00'
        expected_ts = '2013-01-01 01:00:00'
        data = Series([ts] * 3)
        result = to_datetime(data, utc=True, cache=cache)
        expected = Series([Timestamp(expected_ts, tz='utc')] * 3)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('date, dtype', [('2013-01-01 01:00:00', 'datetime64[ns]'), ('2013-01-01 01:00:00', 'datetime64[ns, UTC]')])
    def test_to_datetime_utc_true_with_series_datetime_ns(self, cache, date, dtype):
        expected = Series([Timestamp('2013-01-01 01:00:00', tz='UTC')], dtype='M8[ns, UTC]')
        result = to_datetime(Series([date], dtype=dtype), utc=True, cache=cache)
        tm.assert_series_equal(result, expected)

    def test_to_datetime_tz_psycopg2(self, request, cache):
        psycopg2_tz = pytest.importorskip('psycopg2.tz')
        tz1 = psycopg2_tz.FixedOffsetTimezone(offset=-300, name=None)
        tz2 = psycopg2_tz.FixedOffsetTimezone(offset=-240, name=None)
        arr = np.array([datetime(2000, 1, 1, 3, 0, tzinfo=tz1), datetime(2000, 6, 1, 3, 0, tzinfo=tz2)], dtype=object)
        result = to_datetime(arr, errors='coerce', utc=True, cache=cache)
        expected = DatetimeIndex(['2000-01-01 08:00:00+00:00', '2000-06-01 07:00:00+00:00'], dtype='datetime64[ns, UTC]', freq=None)
        tm.assert_index_equal(result, expected)
        i = DatetimeIndex(['2000-01-01 08:00:00'], tz=psycopg2_tz.FixedOffsetTimezone(offset=-300, name=None))
        assert is_datetime64_ns_dtype(i)
        result = to_datetime(i, errors='coerce', cache=cache)
        tm.assert_index_equal(result, i)
        result = to_datetime(i, errors='coerce', utc=True, cache=cache)
        expected = DatetimeIndex(['2000-01-01 13:00:00'], dtype='datetime64[ns, UTC]')
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('arg', [True, False])
    def test_datetime_bool(self, cache, arg):
        msg = 'dtype bool cannot be converted to datetime64\\[ns\\]'
        with pytest.raises(TypeError, match=msg):
            to_datetime(arg)
        assert to_datetime(arg, errors='coerce', cache=cache) is NaT
        assert to_datetime(arg, errors='ignore', cache=cache) is arg

    def test_datetime_bool_arrays_mixed(self, cache):
        msg = f'{type(cache)} is not convertible to datetime'
        with pytest.raises(TypeError, match=msg):
            to_datetime([False, datetime.today()], cache=cache)
        with pytest.raises(ValueError, match=f"""^time data "True" doesn\\'t match format "%Y%m%d", at position 1. {PARSING_ERR_MSG}$"""):
            to_datetime(['20130101', True], cache=cache)
        tm.assert_index_equal(to_datetime([0, False, NaT, 0.0], errors='coerce', cache=cache), DatetimeIndex([to_datetime(0, cache=cache), NaT, NaT, to_datetime(0, cache=cache)]))

    @pytest.mark.parametrize('arg', [bool, to_datetime])
    def test_datetime_invalid_datatype(self, arg):
        msg = 'is not convertible to datetime'
        with pytest.raises(TypeError, match=msg):
            to_datetime(arg)

    @pytest.mark.parametrize('errors', ['coerce', 'raise', 'ignore'])
    def test_invalid_format_raises(self, errors):
        with pytest.raises(ValueError, match="':' is a bad directive in format 'H%:M%:S%"):
            to_datetime(['00:00:00'], format='H%:M%:S%', errors=errors)

    @pytest.mark.parametrize('value', ['a', '00:01:99'])
    @pytest.mark.parametrize('format', [None, '%H:%M:%S'])
    def test_datetime_invalid_scalar(self, value, format):
        res = to_datetime(value, errors='ignore', format=format)
        assert res == value
        res = to_datetime(value, errors='coerce', format=format)
        assert res is NaT
        msg = '|'.join([f"""^time data "a" doesn\\'t match format "%H:%M:%S", at position 0. {PARSING_ERR_MSG}$""", '^Given date string "a" not likely a datetime, at position 0$', f'^unconverted data remains when parsing with format "%H:%M:%S": "9", at position 0. {PARSING_ERR_MSG}$', '^second must be in 0..59: 00:01:99, at position 0$'])
        with pytest.raises(ValueError, match=msg):
            to_datetime(value, errors='raise', format=format)

    @pytest.mark.parametrize('value', ['3000/12/11 00:00:00'])
    @pytest.mark.parametrize('format', [None, '%H:%M:%S'])
    def test_datetime_outofbounds_scalar(self, value, format):
        res = to_datetime(value, errors='ignore', format=format)
        assert res == value
        res = to_datetime(value, errors='coerce', format=format)
        assert res is NaT
        if format is not None:
            msg = '^time data ".*" doesn\\\'t match format ".*", at position 0.'
            with pytest.raises(ValueError, match=msg):
                to_datetime(value, errors='raise', format=format)
        else:
            msg = '^Out of bounds .*, at position 0$'
            with pytest.raises(OutOfBoundsDatetime, match=msg):
                to_datetime(value, errors='raise', format=format)

    @pytest.mark.parametrize('values', [['a'], ['00:01:99'], ['a', 'b', '99:00:00']])
    @pytest.mark.parametrize('format', [None, '%H:%M:%S'])
    def test_datetime_invalid_index(self, values, format):
        if format is None and len(values) > 1:
            warn = UserWarning
        else:
            warn = None
        with tm.assert_produces_warning(warn, match='Could not infer format', raise_on_extra_warnings=False):
            res = to_datetime(values, errors='ignore', format=format)
        tm.assert_index_equal(res, Index(values, dtype=object))
        with tm.assert_produces_warning(warn, match='Could not infer format', raise_on_extra_warnings=False):
            res = to_datetime(values, errors='coerce', format=format)
        tm.assert_index_equal(res, DatetimeIndex([NaT] * len(values)))
        msg = '|'.join(['^Given date string "a" not likely a datetime, at position 0$', f"""^time data "a" doesn\\'t match format "%H:%M:%S", at position 0. {PARSING_ERR_MSG}$""", f'^unconverted data remains when parsing with format "%H:%M:%S": "9", at position 0. {PARSING_ERR_MSG}$', '^second must be in 0..59: 00:01:99, at position 0$'])
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(warn, match='Could not infer format', raise_on_extra_warnings=False):
                to_datetime(values, errors='raise', format=format)

    @pytest.mark.parametrize('utc', [True, None])
    @pytest.mark.parametrize('format', ['%Y%m%d %H:%M:%S', None])
    @pytest.mark.parametrize('constructor', [list, tuple, np.array, Index, deque])
    def test_to_datetime_cache(self, utc, format, constructor):
        date = '20130101 00:00:00'
        test_dates = [date] * 10 ** 5
        data = constructor(test_dates)
        result = to_datetime(data, utc=utc, format=format, cache=True)
        expected = to_datetime(data, utc=utc, format=format, cache=False)
        tm.assert_index_equal(result, expected)

    def test_to_datetime_from_deque(self):
        result = to_datetime(deque([Timestamp('2010-06-02 09:30:00')] * 51))
        expected = to_datetime([Timestamp('2010-06-02 09:30:00')] * 51)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('utc', [True, None])
    @pytest.mark.parametrize('format', ['%Y%m%d %H:%M:%S', None])
    def test_to_datetime_cache_series(self, utc, format):
        date = '20130101 00:00:00'
        test_dates = [date] * 10 ** 5
        data = Series(test_dates)
        result = to_datetime(data, utc=utc, format=format, cache=True)
        expected = to_datetime(data, utc=utc, format=format, cache=False)
        tm.assert_series_equal(result, expected)

    def test_to_datetime_cache_scalar(self):
        date = '20130101 00:00:00'
        result = to_datetime(date, cache=True)
        expected = Timestamp('20130101 00:00:00')
        assert result == expected

    @pytest.mark.parametrize('datetimelikes,expected_values', (((None, np.nan) + (NaT,) * start_caching_at, (NaT,) * (start_caching_at + 2)), ((None, Timestamp('2012-07-26')) + (NaT,) * start_caching_at, (NaT, Timestamp('2012-07-26')) + (NaT,) * start_caching_at), ((None,) + (NaT,) * start_caching_at + ('2012 July 26', Timestamp('2012-07-26')), (NaT,) * (start_caching_at + 1) + (Timestamp('2012-07-26'), Timestamp('2012-07-26')))))
    def test_convert_object_to_datetime_with_cache(self, datetimelikes, expected_values):
        ser = Series(datetimelikes, dtype='object')
        result_series = to_datetime(ser, errors='coerce')
        expected_series = Series(expected_values, dtype='datetime64[ns]')
        tm.assert_series_equal(result_series, expected_series)

    @pytest.mark.parametrize('cache', [True, False])
    @pytest.mark.parametrize('input', [Series([NaT] * 20 + [None] * 20, dtype='object'), Series([NaT] * 60 + [None] * 60, dtype='object'), Series([None] * 20), Series([None] * 60), Series([''] * 20), Series([''] * 60), Series([pd.NA] * 20), Series([pd.NA] * 60), Series([np.nan] * 20), Series([np.nan] * 60)])
    def test_to_datetime_converts_null_like_to_nat(self, cache, input):
        expected = Series([NaT] * len(input), dtype='M8[ns]')
        result = to_datetime(input, cache=cache)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('date, format', [('2017-20', '%Y-%W'), ('20 Sunday', '%W %A'), ('20 Sun', '%W %a'), ('2017-21', '%Y-%U'), ('20 Sunday', '%U %A'), ('20 Sun', '%U %a')])
    def test_week_without_day_and_calendar_year(self, date, format):
        msg = "Cannot use '%W' or '%U' without day and year"
        with pytest.raises(ValueError, match=msg):
            to_datetime(date, format=format)

    def test_to_datetime_coerce(self):
        ts_strings = ['March 1, 2018 12:00:00+0400', 'March 1, 2018 12:00:00+0500', '20100240']
        msg = 'parsing datetimes with mixed time zones will raise an error'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = to_datetime(ts_strings, errors='coerce')
        expected = Index([datetime(2018, 3, 1, 12, 0, tzinfo=tzoffset(None, 14400)), datetime(2018, 3, 1, 12, 0, tzinfo=tzoffset(None, 18000)), NaT])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('string_arg, format', [('March 1, 2018', '%B %d, %Y'), ('2018-03-01', '%Y-%m-%d')])
    @pytest.mark.parametrize('outofbounds', [datetime(9999, 1, 1), date(9999, 1, 1), np.datetime64('9999-01-01'), 'January 1, 9999', '9999-01-01'])
    def test_to_datetime_coerce_oob(self, string_arg, format, outofbounds):
        ts_strings = [string_arg, outofbounds]
        result = to_datetime(ts_strings, errors='coerce', format=format)
        expected = DatetimeIndex([datetime(2018, 3, 1), NaT])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('errors, expected', [('coerce', Index([NaT, NaT])), ('ignore', Index(['200622-12-31', '111111-24-11'], dtype=object))])
    def test_to_datetime_malformed_no_raise(self, errors, expected):
        ts_strings = ['200622-12-31', '111111-24-11']
        with tm.assert_produces_warning(UserWarning, match='Could not infer format', raise_on_extra_warnings=False):
            result = to_datetime(ts_strings, errors=errors)
        tm.assert_index_equal(result, expected)

    def test_to_datetime_malformed_raise(self):
        ts_strings = ['200622-12-31', '111111-24-11']
        msg = 'Parsed string "200622-12-31" gives an invalid tzoffset, which must be between -timedelta\\(hours=24\\) and timedelta\\(hours=24\\), at position 0'
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(UserWarning, match='Could not infer format'):
                to_datetime(ts_strings, errors='raise')

    def test_iso_8601_strings_with_same_offset(self):
        ts_str = '2015-11-18 15:30:00+05:30'
        result = to_datetime(ts_str)
        expected = Timestamp(ts_str)
        assert result == expected
        expected = DatetimeIndex([Timestamp(ts_str)] * 2)
        result = to_datetime([ts_str] * 2)
        tm.assert_index_equal(result, expected)
        result = DatetimeIndex([ts_str] * 2)
        tm.assert_index_equal(result, expected)

    def test_iso_8601_strings_with_different_offsets(self):
        ts_strings = ['2015-11-18 15:30:00+05:30', '2015-11-18 16:30:00+06:30', NaT]
        msg = 'parsing datetimes with mixed time zones will raise an error'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = to_datetime(ts_strings)
        expected = np.array([datetime(2015, 11, 18, 15, 30, tzinfo=tzoffset(None, 19800)), datetime(2015, 11, 18, 16, 30, tzinfo=tzoffset(None, 23400)), NaT], dtype=object)
        expected = Index(expected)
        tm.assert_index_equal(result, expected)

    def test_iso_8601_strings_with_different_offsets_utc(self):
        ts_strings = ['2015-11-18 15:30:00+05:30', '2015-11-18 16:30:00+06:30', NaT]
        result = to_datetime(ts_strings, utc=True)
        expected = DatetimeIndex([Timestamp(2015, 11, 18, 10), Timestamp(2015, 11, 18, 10), NaT], tz='UTC')
        tm.assert_index_equal(result, expected)

    def test_mixed_offsets_with_native_datetime_raises(self):
        vals = ['nan', Timestamp('1990-01-01'), '2015-03-14T16:15:14.123-08:00', '2019-03-04T21:56:32.620-07:00', None, 'today', 'now']
        ser = Series(vals)
        assert all((ser[i] is vals[i] for i in range(len(vals))))
        now = Timestamp('now')
        today = Timestamp('today')
        msg = 'parsing datetimes with mixed time zones will raise an error'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            mixed = to_datetime(ser)
        expected = Series(['NaT', Timestamp('1990-01-01'), Timestamp('2015-03-14T16:15:14.123-08:00').to_pydatetime(), Timestamp('2019-03-04T21:56:32.620-07:00').to_pydatetime(), None], dtype=object)
        tm.assert_series_equal(mixed[:-2], expected)
        assert (now - mixed.iloc[-1]).total_seconds() <= 0.1
        assert (today - mixed.iloc[-2]).total_seconds() <= 0.1
        with pytest.raises(ValueError, match='Tz-aware datetime.datetime'):
            to_datetime(mixed)

    def test_non_iso_strings_with_tz_offset(self):
        result = to_datetime(['March 1, 2018 12:00:00+0400'] * 2)
        expected = DatetimeIndex([datetime(2018, 3, 1, 12, tzinfo=timezone(timedelta(minutes=240)))] * 2)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('ts, expected', [(Timestamp('2018-01-01'), Timestamp('2018-01-01', tz='UTC')), (Timestamp('2018-01-01', tz='US/Pacific'), Timestamp('2018-01-01 08:00', tz='UTC'))])
    def test_timestamp_utc_true(self, ts, expected):
        result = to_datetime(ts, utc=True)
        assert result == expected

    @pytest.mark.parametrize('dt_str', ['00010101', '13000101', '30000101', '99990101'])
    def test_to_datetime_with_format_out_of_bounds(self, dt_str):
        msg = 'Out of bounds nanosecond timestamp'
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(dt_str, format='%Y%m%d')

    def test_to_datetime_utc(self):
        arr = np.array([parse('2012-06-13T01:39:00Z')], dtype=object)
        result = to_datetime(arr, utc=True)
        assert result.tz is timezone.utc

    def test_to_datetime_fixed_offset(self):
        from pandas.tests.indexes.datetimes.test_timezones import FixedOffset
        fixed_off = FixedOffset(-420, '-07:00')
        dates = [datetime(2000, 1, 1, tzinfo=fixed_off), datetime(2000, 1, 2, tzinfo=fixed_off), datetime(2000, 1, 3, tzinfo=fixed_off)]
        result = to_datetime(dates)
        assert result.tz == fixed_off

    @pytest.mark.parametrize('date', [['2020-10-26 00:00:00+06:00', '2020-10-26 00:00:00+01:00'], ['2020-10-26 00:00:00+06:00', Timestamp('2018-01-01', tz='US/Pacific')], ['2020-10-26 00:00:00+06:00', datetime(2020, 1, 1, 18, tzinfo=pytz.timezone('Australia/Melbourne'))]])
    def test_to_datetime_mixed_offsets_with_utc_false_deprecated(self, date):
        msg = 'parsing datetimes with mixed time zones will raise an error'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            to_datetime(date, utc=False)