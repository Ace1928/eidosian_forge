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
class TestTimeConversionFormats:

    @pytest.mark.parametrize('readonly', [True, False])
    def test_to_datetime_readonly(self, readonly):
        arr = np.array([], dtype=object)
        if readonly:
            arr.setflags(write=False)
        result = to_datetime(arr)
        expected = to_datetime([])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('format, expected', [['%d/%m/%Y', [Timestamp('20000101'), Timestamp('20000201'), Timestamp('20000301')]], ['%m/%d/%Y', [Timestamp('20000101'), Timestamp('20000102'), Timestamp('20000103')]]])
    def test_to_datetime_format(self, cache, index_or_series, format, expected):
        values = index_or_series(['1/1/2000', '1/2/2000', '1/3/2000'])
        result = to_datetime(values, format=format, cache=cache)
        expected = index_or_series(expected)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('arg, expected, format', [['1/1/2000', '20000101', '%d/%m/%Y'], ['1/1/2000', '20000101', '%m/%d/%Y'], ['1/2/2000', '20000201', '%d/%m/%Y'], ['1/2/2000', '20000102', '%m/%d/%Y'], ['1/3/2000', '20000301', '%d/%m/%Y'], ['1/3/2000', '20000103', '%m/%d/%Y']])
    def test_to_datetime_format_scalar(self, cache, arg, expected, format):
        result = to_datetime(arg, format=format, cache=cache)
        expected = Timestamp(expected)
        assert result == expected

    def test_to_datetime_format_YYYYMMDD(self, cache):
        ser = Series([19801222, 19801222] + [19810105] * 5)
        expected = Series([Timestamp(x) for x in ser.apply(str)])
        result = to_datetime(ser, format='%Y%m%d', cache=cache)
        tm.assert_series_equal(result, expected)
        result = to_datetime(ser.apply(str), format='%Y%m%d', cache=cache)
        tm.assert_series_equal(result, expected)

    def test_to_datetime_format_YYYYMMDD_with_nat(self, cache):
        ser = Series([19801222, 19801222] + [19810105] * 5, dtype='float')
        expected = Series([Timestamp('19801222'), Timestamp('19801222')] + [Timestamp('19810105')] * 5)
        expected[2] = np.nan
        ser[2] = np.nan
        result = to_datetime(ser, format='%Y%m%d', cache=cache)
        tm.assert_series_equal(result, expected)
        ser2 = ser.apply(str)
        ser2[2] = 'nat'
        with pytest.raises(ValueError, match='unconverted data remains when parsing with format "%Y%m%d": ".0", at position 0'):
            to_datetime(ser2, format='%Y%m%d', cache=cache)

    def test_to_datetime_format_YYYYMM_with_nat(self, cache):
        ser = Series([198012, 198012] + [198101] * 5, dtype='float')
        expected = Series([Timestamp('19801201'), Timestamp('19801201')] + [Timestamp('19810101')] * 5)
        expected[2] = np.nan
        ser[2] = np.nan
        result = to_datetime(ser, format='%Y%m', cache=cache)
        tm.assert_series_equal(result, expected)

    def test_to_datetime_format_YYYYMMDD_ignore(self, cache):
        ser = Series([20121231, 20141231, 99991231])
        result = to_datetime(ser, format='%Y%m%d', errors='ignore', cache=cache)
        expected = Series([20121231, 20141231, 99991231], dtype=object)
        tm.assert_series_equal(result, expected)

    def test_to_datetime_format_YYYYMMDD_ignore_with_outofbounds(self, cache):
        result = to_datetime(['15010101', '20150101', np.nan], format='%Y%m%d', errors='ignore', cache=cache)
        expected = Index(['15010101', '20150101', np.nan], dtype=object)
        tm.assert_index_equal(result, expected)

    def test_to_datetime_format_YYYYMMDD_coercion(self, cache):
        ser = Series([20121231, 20141231, 99991231])
        result = to_datetime(ser, format='%Y%m%d', errors='coerce', cache=cache)
        expected = Series(['20121231', '20141231', 'NaT'], dtype='M8[ns]')
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('input_s', [['19801222', '20010112', None], ['19801222', '20010112', np.nan], ['19801222', '20010112', NaT], ['19801222', '20010112', 'NaT'], [19801222, 20010112, None], [19801222, 20010112, np.nan], [19801222, 20010112, NaT], [19801222, 20010112, 'NaT']])
    def test_to_datetime_format_YYYYMMDD_with_none(self, input_s):
        expected = Series([Timestamp('19801222'), Timestamp('20010112'), NaT])
        result = Series(to_datetime(input_s, format='%Y%m%d'))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('input_s, expected', [[Series(['19801222', np.nan, '20010012', '10019999']), Series([Timestamp('19801222'), np.nan, np.nan, np.nan])], [Series(['19801222', '20010012', '10019999', np.nan]), Series([Timestamp('19801222'), np.nan, np.nan, np.nan])], [Series([20190813, np.nan, 20010012, 20019999]), Series([Timestamp('20190813'), np.nan, np.nan, np.nan])], [Series([20190813, 20010012, np.nan, 20019999]), Series([Timestamp('20190813'), np.nan, np.nan, np.nan])]])
    def test_to_datetime_format_YYYYMMDD_overflow(self, input_s, expected):
        result = to_datetime(input_s, format='%Y%m%d', errors='coerce')
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('data, format, expected', [([pd.NA], '%Y%m%d%H%M%S', DatetimeIndex(['NaT'])), ([pd.NA], None, DatetimeIndex(['NaT'])), ([pd.NA, '20210202202020'], '%Y%m%d%H%M%S', DatetimeIndex(['NaT', '2021-02-02 20:20:20'])), (['201010', pd.NA], '%y%m%d', DatetimeIndex(['2020-10-10', 'NaT'])), (['201010', pd.NA], '%d%m%y', DatetimeIndex(['2010-10-20', 'NaT'])), ([None, np.nan, pd.NA], None, DatetimeIndex(['NaT', 'NaT', 'NaT'])), ([None, np.nan, pd.NA], '%Y%m%d', DatetimeIndex(['NaT', 'NaT', 'NaT']))])
    def test_to_datetime_with_NA(self, data, format, expected):
        result = to_datetime(data, format=format)
        tm.assert_index_equal(result, expected)

    def test_to_datetime_with_NA_with_warning(self):
        result = to_datetime(['201010', pd.NA])
        expected = DatetimeIndex(['2010-10-20', 'NaT'])
        tm.assert_index_equal(result, expected)

    def test_to_datetime_format_integer(self, cache):
        ser = Series([2000, 2001, 2002])
        expected = Series([Timestamp(x) for x in ser.apply(str)])
        result = to_datetime(ser, format='%Y', cache=cache)
        tm.assert_series_equal(result, expected)
        ser = Series([200001, 200105, 200206])
        expected = Series([Timestamp(x[:4] + '-' + x[4:]) for x in ser.apply(str)])
        result = to_datetime(ser, format='%Y%m', cache=cache)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('int_date, expected', [[20121030, datetime(2012, 10, 30)], [199934, datetime(1999, 3, 4)], [2012010101, 2012010101], [20129930, 20129930], [2012993, 2012993], [2121, 2121]])
    def test_int_to_datetime_format_YYYYMMDD_typeerror(self, int_date, expected):
        result = to_datetime(int_date, format='%Y%m%d', errors='ignore')
        assert result == expected

    def test_to_datetime_format_microsecond(self, cache):
        month_abbr = calendar.month_abbr[4]
        val = f'01-{month_abbr}-2011 00:00:01.978'
        format = '%d-%b-%Y %H:%M:%S.%f'
        result = to_datetime(val, format=format, cache=cache)
        exp = datetime.strptime(val, format)
        assert result == exp

    @pytest.mark.parametrize('value, format, dt', [['01/10/2010 15:20', '%m/%d/%Y %H:%M', Timestamp('2010-01-10 15:20')], ['01/10/2010 05:43', '%m/%d/%Y %I:%M', Timestamp('2010-01-10 05:43')], ['01/10/2010 13:56:01', '%m/%d/%Y %H:%M:%S', Timestamp('2010-01-10 13:56:01')], pytest.param('01/10/2010 08:14 PM', '%m/%d/%Y %I:%M %p', Timestamp('2010-01-10 20:14'), marks=pytest.mark.xfail(locale.getlocale()[0] in ('zh_CN', 'it_IT'), reason='fail on a CI build with LC_ALL=zh_CN.utf8/it_IT.utf8', strict=False)), pytest.param('01/10/2010 07:40 AM', '%m/%d/%Y %I:%M %p', Timestamp('2010-01-10 07:40'), marks=pytest.mark.xfail(locale.getlocale()[0] in ('zh_CN', 'it_IT'), reason='fail on a CI build with LC_ALL=zh_CN.utf8/it_IT.utf8', strict=False)), pytest.param('01/10/2010 09:12:56 AM', '%m/%d/%Y %I:%M:%S %p', Timestamp('2010-01-10 09:12:56'), marks=pytest.mark.xfail(locale.getlocale()[0] in ('zh_CN', 'it_IT'), reason='fail on a CI build with LC_ALL=zh_CN.utf8/it_IT.utf8', strict=False))])
    def test_to_datetime_format_time(self, cache, value, format, dt):
        assert to_datetime(value, format=format, cache=cache) == dt

    @td.skip_if_not_us_locale
    def test_to_datetime_with_non_exact(self, cache):
        ser = Series(['19MAY11', 'foobar19MAY11', '19MAY11:00:00:00', '19MAY11 00:00:00Z'])
        result = to_datetime(ser, format='%d%b%y', exact=False, cache=cache)
        expected = to_datetime(ser.str.extract('(\\d+\\w+\\d+)', expand=False), format='%d%b%y', cache=cache)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('format, expected', [('%Y-%m-%d', Timestamp(2000, 1, 3)), ('%Y-%d-%m', Timestamp(2000, 3, 1)), ('%Y-%m-%d %H', Timestamp(2000, 1, 3, 12)), ('%Y-%d-%m %H', Timestamp(2000, 3, 1, 12)), ('%Y-%m-%d %H:%M', Timestamp(2000, 1, 3, 12, 34)), ('%Y-%d-%m %H:%M', Timestamp(2000, 3, 1, 12, 34)), ('%Y-%m-%d %H:%M:%S', Timestamp(2000, 1, 3, 12, 34, 56)), ('%Y-%d-%m %H:%M:%S', Timestamp(2000, 3, 1, 12, 34, 56)), ('%Y-%m-%d %H:%M:%S.%f', Timestamp(2000, 1, 3, 12, 34, 56, 123456)), ('%Y-%d-%m %H:%M:%S.%f', Timestamp(2000, 3, 1, 12, 34, 56, 123456)), ('%Y-%m-%d %H:%M:%S.%f%z', Timestamp(2000, 1, 3, 12, 34, 56, 123456, tz='UTC+01:00')), ('%Y-%d-%m %H:%M:%S.%f%z', Timestamp(2000, 3, 1, 12, 34, 56, 123456, tz='UTC+01:00'))])
    def test_non_exact_doesnt_parse_whole_string(self, cache, format, expected):
        result = to_datetime('2000-01-03 12:34:56.123456+01:00', format=format, exact=False)
        assert result == expected

    @pytest.mark.parametrize('arg', ['2012-01-01 09:00:00.000000001', '2012-01-01 09:00:00.000001', '2012-01-01 09:00:00.001', '2012-01-01 09:00:00.001000', '2012-01-01 09:00:00.001000000'])
    def test_parse_nanoseconds_with_formula(self, cache, arg):
        expected = to_datetime(arg, cache=cache)
        result = to_datetime(arg, format='%Y-%m-%d %H:%M:%S.%f', cache=cache)
        assert result == expected

    @pytest.mark.parametrize('value,fmt,expected', [['2009324', '%Y%W%w', Timestamp('2009-08-13')], ['2013020', '%Y%U%w', Timestamp('2013-01-13')]])
    def test_to_datetime_format_weeks(self, value, fmt, expected, cache):
        assert to_datetime(value, format=fmt, cache=cache) == expected

    @pytest.mark.parametrize('fmt,dates,expected_dates', [['%Y-%m-%d %H:%M:%S %Z', ['2010-01-01 12:00:00 UTC'] * 2, [Timestamp('2010-01-01 12:00:00', tz='UTC')] * 2], ['%Y-%m-%d %H:%M:%S%z', ['2010-01-01 12:00:00+0100'] * 2, [Timestamp('2010-01-01 12:00:00', tzinfo=timezone(timedelta(minutes=60)))] * 2], ['%Y-%m-%d %H:%M:%S %z', ['2010-01-01 12:00:00 +0100'] * 2, [Timestamp('2010-01-01 12:00:00', tzinfo=timezone(timedelta(minutes=60)))] * 2], ['%Y-%m-%d %H:%M:%S %z', ['2010-01-01 12:00:00 Z', '2010-01-01 12:00:00 Z'], [Timestamp('2010-01-01 12:00:00', tzinfo=pytz.FixedOffset(0)), Timestamp('2010-01-01 12:00:00', tzinfo=pytz.FixedOffset(0))]]])
    def test_to_datetime_parse_tzname_or_tzoffset(self, fmt, dates, expected_dates):
        result = to_datetime(dates, format=fmt)
        expected = Index(expected_dates)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('fmt,dates,expected_dates', [['%Y-%m-%d %H:%M:%S %Z', ['2010-01-01 12:00:00 UTC', '2010-01-01 12:00:00 GMT', '2010-01-01 12:00:00 US/Pacific'], [Timestamp('2010-01-01 12:00:00', tz='UTC'), Timestamp('2010-01-01 12:00:00', tz='GMT'), Timestamp('2010-01-01 12:00:00', tz='US/Pacific')]], ['%Y-%m-%d %H:%M:%S %z', ['2010-01-01 12:00:00 +0100', '2010-01-01 12:00:00 -0100'], [Timestamp('2010-01-01 12:00:00', tzinfo=timezone(timedelta(minutes=60))), Timestamp('2010-01-01 12:00:00', tzinfo=timezone(timedelta(minutes=-60)))]]])
    def test_to_datetime_parse_tzname_or_tzoffset_utc_false_deprecated(self, fmt, dates, expected_dates):
        msg = 'parsing datetimes with mixed time zones will raise an error'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = to_datetime(dates, format=fmt)
        expected = Index(expected_dates)
        tm.assert_equal(result, expected)

    def test_to_datetime_parse_tzname_or_tzoffset_different_tz_to_utc(self):
        dates = ['2010-01-01 12:00:00 +0100', '2010-01-01 12:00:00 -0100', '2010-01-01 12:00:00 +0300', '2010-01-01 12:00:00 +0400']
        expected_dates = ['2010-01-01 11:00:00+00:00', '2010-01-01 13:00:00+00:00', '2010-01-01 09:00:00+00:00', '2010-01-01 08:00:00+00:00']
        fmt = '%Y-%m-%d %H:%M:%S %z'
        result = to_datetime(dates, format=fmt, utc=True)
        expected = DatetimeIndex(expected_dates)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('offset', ['+0', '-1foo', 'UTCbar', ':10', '+01:000:01', ''])
    def test_to_datetime_parse_timezone_malformed(self, offset):
        fmt = '%Y-%m-%d %H:%M:%S %z'
        date = '2010-01-01 12:00:00 ' + offset
        msg = '|'.join([f"""^time data ".*" doesn\\'t match format ".*", at position 0. {PARSING_ERR_MSG}$""", f'^unconverted data remains when parsing with format ".*": ".*", at position 0. {PARSING_ERR_MSG}$'])
        with pytest.raises(ValueError, match=msg):
            to_datetime([date], format=fmt)

    def test_to_datetime_parse_timezone_keeps_name(self):
        fmt = '%Y-%m-%d %H:%M:%S %z'
        arg = Index(['2010-01-01 12:00:00 Z'], name='foo')
        result = to_datetime(arg, format=fmt)
        expected = DatetimeIndex(['2010-01-01 12:00:00'], tz='UTC', name='foo')
        tm.assert_index_equal(result, expected)