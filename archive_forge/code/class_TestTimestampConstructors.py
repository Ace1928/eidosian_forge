import calendar
from datetime import (
import zoneinfo
import dateutil.tz
from dateutil.tz import (
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.compat import PY310
from pandas.errors import OutOfBoundsDatetime
from pandas import (
class TestTimestampConstructors:

    def test_weekday_but_no_day_raises(self):
        msg = 'Parsing datetimes with weekday but no day information is not supported'
        with pytest.raises(ValueError, match=msg):
            Timestamp('2023 Sept Thu')

    def test_construct_from_string_invalid_raises(self):
        with pytest.raises(ValueError, match='gives an invalid tzoffset'):
            Timestamp('200622-12-31')

    def test_constructor_from_iso8601_str_with_offset_reso(self):
        ts = Timestamp('2016-01-01 04:05:06-01:00')
        assert ts.unit == 's'
        ts = Timestamp('2016-01-01 04:05:06.000-01:00')
        assert ts.unit == 'ms'
        ts = Timestamp('2016-01-01 04:05:06.000000-01:00')
        assert ts.unit == 'us'
        ts = Timestamp('2016-01-01 04:05:06.000000001-01:00')
        assert ts.unit == 'ns'

    def test_constructor_from_date_second_reso(self):
        obj = date(2012, 9, 1)
        ts = Timestamp(obj)
        assert ts.unit == 's'

    def test_constructor_datetime64_with_tz(self):
        dt = np.datetime64('1970-01-01 05:00:00')
        tzstr = 'UTC+05:00'
        ts = Timestamp(dt, tz=tzstr)
        alt = Timestamp(dt).tz_localize(tzstr)
        assert ts == alt
        assert ts.hour == 5

    def test_constructor(self):
        base_str = '2014-07-01 09:00'
        base_dt = datetime(2014, 7, 1, 9)
        base_expected = 1404205200000000000
        assert calendar.timegm(base_dt.timetuple()) * 1000000000 == base_expected
        tests = [(base_str, base_dt, base_expected), ('2014-07-01 10:00', datetime(2014, 7, 1, 10), base_expected + 3600 * 1000000000), ('2014-07-01 09:00:00.000008000', datetime(2014, 7, 1, 9, 0, 0, 8), base_expected + 8000), ('2014-07-01 09:00:00.000000005', Timestamp('2014-07-01 09:00:00.000000005'), base_expected + 5)]
        timezones = [(None, 0), ('UTC', 0), (pytz.utc, 0), ('Asia/Tokyo', 9), ('US/Eastern', -4), ('dateutil/US/Pacific', -7), (pytz.FixedOffset(-180), -3), (dateutil.tz.tzoffset(None, 18000), 5)]
        for date_str, date_obj, expected in tests:
            for result in [Timestamp(date_str), Timestamp(date_obj)]:
                result = result.as_unit('ns')
                assert result.as_unit('ns')._value == expected
                result = Timestamp(result)
                assert result.as_unit('ns')._value == expected
            for tz, offset in timezones:
                for result in [Timestamp(date_str, tz=tz), Timestamp(date_obj, tz=tz)]:
                    result = result.as_unit('ns')
                    expected_tz = expected - offset * 3600 * 1000000000
                    assert result.as_unit('ns')._value == expected_tz
                    result = Timestamp(result)
                    assert result.as_unit('ns')._value == expected_tz
                    if tz is not None:
                        result = Timestamp(result).tz_convert('UTC')
                    else:
                        result = Timestamp(result, tz='UTC')
                    expected_utc = expected - offset * 3600 * 1000000000
                    assert result.as_unit('ns')._value == expected_utc

    def test_constructor_with_stringoffset(self):
        base_str = '2014-07-01 11:00:00+02:00'
        base_dt = datetime(2014, 7, 1, 9)
        base_expected = 1404205200000000000
        assert calendar.timegm(base_dt.timetuple()) * 1000000000 == base_expected
        tests = [(base_str, base_expected), ('2014-07-01 12:00:00+02:00', base_expected + 3600 * 1000000000), ('2014-07-01 11:00:00.000008000+02:00', base_expected + 8000), ('2014-07-01 11:00:00.000000005+02:00', base_expected + 5)]
        timezones = [(None, 0), ('UTC', 0), (pytz.utc, 0), ('Asia/Tokyo', 9), ('US/Eastern', -4), ('dateutil/US/Pacific', -7), (pytz.FixedOffset(-180), -3), (dateutil.tz.tzoffset(None, 18000), 5)]
        for date_str, expected in tests:
            for result in [Timestamp(date_str)]:
                assert result.as_unit('ns')._value == expected
                result = Timestamp(result)
                assert result.as_unit('ns')._value == expected
            for tz, offset in timezones:
                result = Timestamp(date_str, tz=tz)
                expected_tz = expected
                assert result.as_unit('ns')._value == expected_tz
                result = Timestamp(result)
                assert result.as_unit('ns')._value == expected_tz
                result = Timestamp(result).tz_convert('UTC')
                expected_utc = expected
                assert result.as_unit('ns')._value == expected_utc
        result = Timestamp('2013-11-01 00:00:00-0500', tz='America/Chicago')
        assert result._value == Timestamp('2013-11-01 05:00')._value
        expected = "Timestamp('2013-11-01 00:00:00-0500', tz='America/Chicago')"
        assert repr(result) == expected
        assert result == eval(repr(result))
        result = Timestamp('2013-11-01 00:00:00-0500', tz='Asia/Tokyo')
        assert result._value == Timestamp('2013-11-01 05:00')._value
        expected = "Timestamp('2013-11-01 14:00:00+0900', tz='Asia/Tokyo')"
        assert repr(result) == expected
        assert result == eval(repr(result))
        result = Timestamp('2015-11-18 15:45:00+05:45', tz='Asia/Katmandu')
        assert result._value == Timestamp('2015-11-18 10:00')._value
        expected = "Timestamp('2015-11-18 15:45:00+0545', tz='Asia/Katmandu')"
        assert repr(result) == expected
        assert result == eval(repr(result))
        result = Timestamp('2015-11-18 15:30:00+05:30', tz='Asia/Kolkata')
        assert result._value == Timestamp('2015-11-18 10:00')._value
        expected = "Timestamp('2015-11-18 15:30:00+0530', tz='Asia/Kolkata')"
        assert repr(result) == expected
        assert result == eval(repr(result))

    def test_constructor_invalid(self):
        msg = 'Cannot convert input'
        with pytest.raises(TypeError, match=msg):
            Timestamp(slice(2))
        msg = 'Cannot convert Period'
        with pytest.raises(ValueError, match=msg):
            Timestamp(Period('1000-01-01'))

    def test_constructor_invalid_tz(self):
        msg = "Argument 'tzinfo' has incorrect type \\(expected datetime.tzinfo, got str\\)"
        with pytest.raises(TypeError, match=msg):
            Timestamp('2017-10-22', tzinfo='US/Eastern')
        msg = 'at most one of'
        with pytest.raises(ValueError, match=msg):
            Timestamp('2017-10-22', tzinfo=pytz.utc, tz='UTC')
        msg = 'Cannot pass a date attribute keyword argument when passing a date string'
        with pytest.raises(ValueError, match=msg):
            Timestamp('2012-01-01', 'US/Pacific')

    def test_constructor_tz_or_tzinfo(self):
        stamps = [Timestamp(year=2017, month=10, day=22, tz='UTC'), Timestamp(year=2017, month=10, day=22, tzinfo=pytz.utc), Timestamp(year=2017, month=10, day=22, tz=pytz.utc), Timestamp(datetime(2017, 10, 22), tzinfo=pytz.utc), Timestamp(datetime(2017, 10, 22), tz='UTC'), Timestamp(datetime(2017, 10, 22), tz=pytz.utc)]
        assert all((ts == stamps[0] for ts in stamps))

    @pytest.mark.parametrize('result', [Timestamp(datetime(2000, 1, 2, 3, 4, 5, 6), nanosecond=1), Timestamp(year=2000, month=1, day=2, hour=3, minute=4, second=5, microsecond=6, nanosecond=1), Timestamp(year=2000, month=1, day=2, hour=3, minute=4, second=5, microsecond=6, nanosecond=1, tz='UTC'), Timestamp(2000, 1, 2, 3, 4, 5, 6, None, nanosecond=1), Timestamp(2000, 1, 2, 3, 4, 5, 6, tz=pytz.UTC, nanosecond=1)])
    def test_constructor_nanosecond(self, result):
        expected = Timestamp(datetime(2000, 1, 2, 3, 4, 5, 6), tz=result.tz)
        expected = expected + Timedelta(nanoseconds=1)
        assert result == expected

    @pytest.mark.parametrize('z', ['Z0', 'Z00'])
    def test_constructor_invalid_Z0_isostring(self, z):
        msg = f'Unknown datetime string format, unable to parse: 2014-11-02 01:00{z}'
        with pytest.raises(ValueError, match=msg):
            Timestamp(f'2014-11-02 01:00{z}')

    def test_out_of_bounds_integer_value(self):
        msg = str(Timestamp.max._value * 2)
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp(Timestamp.max._value * 2)
        msg = str(Timestamp.min._value * 2)
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp(Timestamp.min._value * 2)

    def test_out_of_bounds_value(self):
        one_us = np.timedelta64(1).astype('timedelta64[us]')
        min_ts_us = np.datetime64(Timestamp.min).astype('M8[us]') + one_us
        max_ts_us = np.datetime64(Timestamp.max).astype('M8[us]')
        Timestamp(min_ts_us)
        Timestamp(max_ts_us)
        us_val = NpyDatetimeUnit.NPY_FR_us.value
        assert Timestamp(min_ts_us - one_us)._creso == us_val
        assert Timestamp(max_ts_us + one_us)._creso == us_val
        too_low = np.datetime64('-292277022657-01-27T08:29', 'm')
        too_high = np.datetime64('292277026596-12-04T15:31', 'm')
        msg = 'Out of bounds'
        with pytest.raises(ValueError, match=msg):
            Timestamp(too_low)
        with pytest.raises(ValueError, match=msg):
            Timestamp(too_high)

    def test_out_of_bounds_string(self):
        msg = "Cannot cast .* to unit='ns' without overflow"
        with pytest.raises(ValueError, match=msg):
            Timestamp('1676-01-01').as_unit('ns')
        with pytest.raises(ValueError, match=msg):
            Timestamp('2263-01-01').as_unit('ns')
        ts = Timestamp('2263-01-01')
        assert ts.unit == 's'
        ts = Timestamp('1676-01-01')
        assert ts.unit == 's'

    def test_barely_out_of_bounds(self):
        msg = 'Out of bounds nanosecond timestamp: 2262-04-11 23:47:16'
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp('2262-04-11 23:47:16.854775808')

    @pytest.mark.skip_ubsan
    def test_bounds_with_different_units(self):
        out_of_bounds_dates = ('1677-09-21', '2262-04-12')
        time_units = ('D', 'h', 'm', 's', 'ms', 'us')
        for date_string in out_of_bounds_dates:
            for unit in time_units:
                dt64 = np.datetime64(date_string, unit)
                ts = Timestamp(dt64)
                if unit in ['s', 'ms', 'us']:
                    assert ts._value == dt64.view('i8')
                else:
                    assert ts._creso == NpyDatetimeUnit.NPY_FR_s.value
        info = np.iinfo(np.int64)
        msg = 'Out of bounds second timestamp:'
        for value in [info.min + 1, info.max]:
            for unit in ['D', 'h', 'm']:
                dt64 = np.datetime64(value, unit)
                with pytest.raises(OutOfBoundsDatetime, match=msg):
                    Timestamp(dt64)
        in_bounds_dates = ('1677-09-23', '2262-04-11')
        for date_string in in_bounds_dates:
            for unit in time_units:
                dt64 = np.datetime64(date_string, unit)
                Timestamp(dt64)

    @pytest.mark.parametrize('arg', ['001-01-01', '0001-01-01'])
    def test_out_of_bounds_string_consistency(self, arg):
        msg = "Cannot cast 0001-01-01 00:00:00 to unit='ns' without overflow"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp(arg).as_unit('ns')
        ts = Timestamp(arg)
        assert ts.unit == 's'
        assert ts.year == ts.month == ts.day == 1

    def test_min_valid(self):
        Timestamp(Timestamp.min)

    def test_max_valid(self):
        Timestamp(Timestamp.max)

    @pytest.mark.parametrize('offset', ['+0300', '+0200'])
    def test_construct_timestamp_near_dst(self, offset):
        expected = Timestamp(f'2016-10-30 03:00:00{offset}', tz='Europe/Helsinki')
        result = Timestamp(expected).tz_convert('Europe/Helsinki')
        assert result == expected

    @pytest.mark.parametrize('arg', ['2013/01/01 00:00:00+09:00', '2013-01-01 00:00:00+09:00'])
    def test_construct_with_different_string_format(self, arg):
        result = Timestamp(arg)
        expected = Timestamp(datetime(2013, 1, 1), tz=pytz.FixedOffset(540))
        assert result == expected

    @pytest.mark.parametrize('box', [datetime, Timestamp])
    def test_raise_tz_and_tzinfo_in_datetime_input(self, box):
        kwargs = {'year': 2018, 'month': 1, 'day': 1, 'tzinfo': pytz.utc}
        msg = 'Cannot pass a datetime or Timestamp'
        with pytest.raises(ValueError, match=msg):
            Timestamp(box(**kwargs), tz='US/Pacific')
        msg = 'Cannot pass a datetime or Timestamp'
        with pytest.raises(ValueError, match=msg):
            Timestamp(box(**kwargs), tzinfo=pytz.timezone('US/Pacific'))

    def test_dont_convert_dateutil_utc_to_pytz_utc(self):
        result = Timestamp(datetime(2018, 1, 1), tz=tzutc())
        expected = Timestamp(datetime(2018, 1, 1)).tz_localize(tzutc())
        assert result == expected

    def test_constructor_subclassed_datetime(self):

        class SubDatetime(datetime):
            pass
        data = SubDatetime(2000, 1, 1)
        result = Timestamp(data)
        expected = Timestamp(2000, 1, 1)
        assert result == expected

    def test_timestamp_constructor_tz_utc(self):
        utc_stamp = Timestamp('3/11/2012 05:00', tz='utc')
        assert utc_stamp.tzinfo is timezone.utc
        assert utc_stamp.hour == 5
        utc_stamp = Timestamp('3/11/2012 05:00').tz_localize('utc')
        assert utc_stamp.hour == 5

    def test_timestamp_to_datetime_tzoffset(self):
        tzinfo = tzoffset(None, 7200)
        expected = Timestamp('3/11/2012 04:00', tz=tzinfo)
        result = Timestamp(expected.to_pydatetime())
        assert expected == result

    def test_timestamp_constructor_near_dst_boundary(self):
        for tz in ['Europe/Brussels', 'Europe/Prague']:
            result = Timestamp('2015-10-25 01:00', tz=tz)
            expected = Timestamp('2015-10-25 01:00').tz_localize(tz)
            assert result == expected
            msg = 'Cannot infer dst time from 2015-10-25 02:00:00'
            with pytest.raises(pytz.AmbiguousTimeError, match=msg):
                Timestamp('2015-10-25 02:00', tz=tz)
        result = Timestamp('2017-03-26 01:00', tz='Europe/Paris')
        expected = Timestamp('2017-03-26 01:00').tz_localize('Europe/Paris')
        assert result == expected
        msg = '2017-03-26 02:00'
        with pytest.raises(pytz.NonExistentTimeError, match=msg):
            Timestamp('2017-03-26 02:00', tz='Europe/Paris')
        naive = Timestamp('2015-11-18 10:00:00')
        result = naive.tz_localize('UTC').tz_convert('Asia/Kolkata')
        expected = Timestamp('2015-11-18 15:30:00+0530', tz='Asia/Kolkata')
        assert result == expected
        result = Timestamp('2017-03-26 00:00', tz='Europe/Paris')
        expected = Timestamp('2017-03-26 00:00:00+0100', tz='Europe/Paris')
        assert result == expected
        result = Timestamp('2017-03-26 01:00', tz='Europe/Paris')
        expected = Timestamp('2017-03-26 01:00:00+0100', tz='Europe/Paris')
        assert result == expected
        msg = '2017-03-26 02:00'
        with pytest.raises(pytz.NonExistentTimeError, match=msg):
            Timestamp('2017-03-26 02:00', tz='Europe/Paris')
        result = Timestamp('2017-03-26 02:00:00+0100', tz='Europe/Paris')
        naive = Timestamp(result.as_unit('ns')._value)
        expected = naive.tz_localize('UTC').tz_convert('Europe/Paris')
        assert result == expected
        result = Timestamp('2017-03-26 03:00', tz='Europe/Paris')
        expected = Timestamp('2017-03-26 03:00:00+0200', tz='Europe/Paris')
        assert result == expected

    @pytest.mark.parametrize('tz', [pytz.timezone('US/Eastern'), gettz('US/Eastern'), 'US/Eastern', 'dateutil/US/Eastern'])
    def test_timestamp_constructed_by_date_and_tz(self, tz):
        result = Timestamp(date(2012, 3, 11), tz=tz)
        expected = Timestamp('3/11/2012', tz=tz)
        assert result.hour == expected.hour
        assert result == expected