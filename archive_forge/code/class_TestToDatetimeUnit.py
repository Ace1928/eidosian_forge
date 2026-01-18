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
class TestToDatetimeUnit:

    @pytest.mark.parametrize('unit', ['Y', 'M'])
    @pytest.mark.parametrize('item', [150, float(150)])
    def test_to_datetime_month_or_year_unit_int(self, cache, unit, item, request):
        ts = Timestamp(item, unit=unit)
        expected = DatetimeIndex([ts], dtype='M8[ns]')
        result = to_datetime([item], unit=unit, cache=cache)
        tm.assert_index_equal(result, expected)
        result = to_datetime(np.array([item], dtype=object), unit=unit, cache=cache)
        tm.assert_index_equal(result, expected)
        result = to_datetime(np.array([item]), unit=unit, cache=cache)
        tm.assert_index_equal(result, expected)
        result = to_datetime(np.array([item, np.nan]), unit=unit, cache=cache)
        assert result.isna()[1]
        tm.assert_index_equal(result[:1], expected)

    @pytest.mark.parametrize('unit', ['Y', 'M'])
    def test_to_datetime_month_or_year_unit_non_round_float(self, cache, unit):
        warn_msg = 'strings will be parsed as datetime strings'
        msg = f'Conversion of non-round float with unit={unit} is ambiguous'
        with pytest.raises(ValueError, match=msg):
            to_datetime([1.5], unit=unit, errors='raise')
        with pytest.raises(ValueError, match=msg):
            to_datetime(np.array([1.5]), unit=unit, errors='raise')
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(FutureWarning, match=warn_msg):
                to_datetime(['1.5'], unit=unit, errors='raise')
        with pytest.raises(ValueError, match=msg):
            to_datetime([1.5], unit=unit, errors='ignore')
        res = to_datetime([1.5], unit=unit, errors='coerce')
        expected = Index([NaT], dtype='M8[ns]')
        tm.assert_index_equal(res, expected)
        with tm.assert_produces_warning(FutureWarning, match=warn_msg):
            res = to_datetime(['1.5'], unit=unit, errors='coerce')
        tm.assert_index_equal(res, expected)
        res = to_datetime([1.0], unit=unit)
        expected = to_datetime([1], unit=unit)
        tm.assert_index_equal(res, expected)

    def test_unit(self, cache):
        msg = 'cannot specify both format and unit'
        with pytest.raises(ValueError, match=msg):
            to_datetime([1], unit='D', format='%Y%m%d', cache=cache)

    def test_unit_array_mixed_nans(self, cache):
        values = [11111111111111111, 1, 1.0, iNaT, NaT, np.nan, 'NaT', '']
        result = to_datetime(values, unit='D', errors='ignore', cache=cache)
        expected = Index([11111111111111111, Timestamp('1970-01-02'), Timestamp('1970-01-02'), NaT, NaT, NaT, NaT, NaT], dtype=object)
        tm.assert_index_equal(result, expected)
        result = to_datetime(values, unit='D', errors='coerce', cache=cache)
        expected = DatetimeIndex(['NaT', '1970-01-02', '1970-01-02', 'NaT', 'NaT', 'NaT', 'NaT', 'NaT'], dtype='M8[ns]')
        tm.assert_index_equal(result, expected)
        msg = "cannot convert input 11111111111111111 with the unit 'D'"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(values, unit='D', errors='raise', cache=cache)

    def test_unit_array_mixed_nans_large_int(self, cache):
        values = [1420043460000000000000000, iNaT, NaT, np.nan, 'NaT']
        result = to_datetime(values, errors='ignore', unit='s', cache=cache)
        expected = Index([1420043460000000000000000, NaT, NaT, NaT, NaT], dtype=object)
        tm.assert_index_equal(result, expected)
        result = to_datetime(values, errors='coerce', unit='s', cache=cache)
        expected = DatetimeIndex(['NaT', 'NaT', 'NaT', 'NaT', 'NaT'], dtype='M8[ns]')
        tm.assert_index_equal(result, expected)
        msg = "cannot convert input 1420043460000000000000000 with the unit 's'"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(values, errors='raise', unit='s', cache=cache)

    def test_to_datetime_invalid_str_not_out_of_bounds_valuerror(self, cache):
        msg = "non convertible value foo with the unit 's'"
        with pytest.raises(ValueError, match=msg):
            to_datetime('foo', errors='raise', unit='s', cache=cache)

    @pytest.mark.parametrize('error', ['raise', 'coerce', 'ignore'])
    def test_unit_consistency(self, cache, error):
        expected = Timestamp('1970-05-09 14:25:11')
        result = to_datetime(11111111, unit='s', errors=error, cache=cache)
        assert result == expected
        assert isinstance(result, Timestamp)

    @pytest.mark.parametrize('errors', ['ignore', 'raise', 'coerce'])
    @pytest.mark.parametrize('dtype', ['float64', 'int64'])
    def test_unit_with_numeric(self, cache, errors, dtype):
        expected = DatetimeIndex(['2015-06-19 05:33:20', '2015-05-27 22:33:20'], dtype='M8[ns]')
        arr = np.array([1.434692e+18, 1.432766e+18]).astype(dtype)
        result = to_datetime(arr, errors=errors, cache=cache)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('exp, arr, warning', [[['NaT', '2015-06-19 05:33:20', '2015-05-27 22:33:20'], ['foo', 1.434692e+18, 1.432766e+18], UserWarning], [['2015-06-19 05:33:20', '2015-05-27 22:33:20', 'NaT', 'NaT'], [1.434692e+18, 1.432766e+18, 'foo', 'NaT'], None]])
    def test_unit_with_numeric_coerce(self, cache, exp, arr, warning):
        expected = DatetimeIndex(exp, dtype='M8[ns]')
        with tm.assert_produces_warning(warning, match='Could not infer format'):
            result = to_datetime(arr, errors='coerce', cache=cache)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('arr', [[Timestamp('20130101'), 1.434692e+18, 1.432766e+18], [1.434692e+18, 1.432766e+18, Timestamp('20130101')]])
    def test_unit_mixed(self, cache, arr):
        expected = Index([Timestamp(x) for x in arr], dtype='M8[ns]')
        result = to_datetime(arr, errors='coerce', cache=cache)
        tm.assert_index_equal(result, expected)
        result = to_datetime(arr, errors='raise', cache=cache)
        tm.assert_index_equal(result, expected)
        result = DatetimeIndex(arr)
        tm.assert_index_equal(result, expected)

    def test_unit_rounding(self, cache):
        value = 1434743731.877
        result = to_datetime(value, unit='s', cache=cache)
        expected = Timestamp('2015-06-19 19:55:31.877000093')
        assert result == expected
        alt = Timestamp(value, unit='s')
        assert alt == result

    def test_unit_ignore_keeps_name(self, cache):
        expected = Index([15000000000.0] * 2, name='name')
        result = to_datetime(expected, errors='ignore', unit='s', cache=cache)
        tm.assert_index_equal(result, expected)

    def test_to_datetime_errors_ignore_utc_true(self):
        result = to_datetime([1], unit='s', utc=True, errors='ignore')
        expected = DatetimeIndex(['1970-01-01 00:00:01'], dtype='M8[ns, UTC]')
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('dtype', [int, float])
    def test_to_datetime_unit(self, dtype):
        epoch = 1370745748
        ser = Series([epoch + t for t in range(20)]).astype(dtype)
        result = to_datetime(ser, unit='s')
        expected = Series([Timestamp('2013-06-09 02:42:28') + timedelta(seconds=t) for t in range(20)], dtype='M8[ns]')
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('null', [iNaT, np.nan])
    def test_to_datetime_unit_with_nulls(self, null):
        epoch = 1370745748
        ser = Series([epoch + t for t in range(20)] + [null])
        result = to_datetime(ser, unit='s')
        expected = Series([Timestamp('2013-06-09 02:42:28') + timedelta(seconds=t) for t in range(20)] + [NaT], dtype='M8[ns]')
        tm.assert_series_equal(result, expected)

    def test_to_datetime_unit_fractional_seconds(self):
        epoch = 1370745748
        ser = Series([epoch + t for t in np.arange(0, 2, 0.25)] + [iNaT]).astype(float)
        result = to_datetime(ser, unit='s')
        expected = Series([Timestamp('2013-06-09 02:42:28') + timedelta(seconds=t) for t in np.arange(0, 2, 0.25)] + [NaT], dtype='M8[ns]')
        result = result.round('ms')
        tm.assert_series_equal(result, expected)

    def test_to_datetime_unit_na_values(self):
        result = to_datetime([1, 2, 'NaT', NaT, np.nan], unit='D')
        expected = DatetimeIndex([Timestamp('1970-01-02'), Timestamp('1970-01-03')] + ['NaT'] * 3, dtype='M8[ns]')
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('bad_val', ['foo', 111111111])
    def test_to_datetime_unit_invalid(self, bad_val):
        msg = f"{bad_val} with the unit 'D'"
        with pytest.raises(ValueError, match=msg):
            to_datetime([1, 2, bad_val], unit='D')

    @pytest.mark.parametrize('bad_val', ['foo', 111111111])
    def test_to_timestamp_unit_coerce(self, bad_val):
        expected = DatetimeIndex([Timestamp('1970-01-02'), Timestamp('1970-01-03')] + ['NaT'] * 1, dtype='M8[ns]')
        result = to_datetime([1, 2, bad_val], unit='D', errors='coerce')
        tm.assert_index_equal(result, expected)

    def test_float_to_datetime_raise_near_bounds(self):
        msg = "cannot convert input with unit 'D'"
        oneday_in_ns = 1000000000.0 * 60 * 60 * 24
        tsmax_in_days = 2 ** 63 / oneday_in_ns
        should_succeed = Series([0, tsmax_in_days - 0.005, -tsmax_in_days + 0.005], dtype=float)
        expected = (should_succeed * oneday_in_ns).astype(np.int64)
        for error_mode in ['raise', 'coerce', 'ignore']:
            result1 = to_datetime(should_succeed, unit='D', errors=error_mode)
            tm.assert_almost_equal(result1.astype(np.int64).astype(np.float64), expected.astype(np.float64), rtol=1e-10)
        should_fail1 = Series([0, tsmax_in_days + 0.005], dtype=float)
        should_fail2 = Series([0, -tsmax_in_days - 0.005], dtype=float)
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(should_fail1, unit='D', errors='raise')
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            to_datetime(should_fail2, unit='D', errors='raise')