from datetime import (
from itertools import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.conversion import localize_pydatetime
from pandas._libs.tslibs.offsets import shift_months
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.tests.arithmetic.common import (
class TestTimestampSeriesArithmetic:

    def test_empty_series_add_sub(self, box_with_array):
        a = Series(dtype='M8[ns]')
        b = Series(dtype='m8[ns]')
        a = box_with_array(a)
        b = box_with_array(b)
        tm.assert_equal(a, a + b)
        tm.assert_equal(a, a - b)
        tm.assert_equal(a, b + a)
        msg = 'cannot subtract'
        with pytest.raises(TypeError, match=msg):
            b - a

    def test_operators_datetimelike(self):
        td1 = Series([timedelta(minutes=5, seconds=3)] * 3)
        td1.iloc[2] = np.nan
        dt1 = Series([Timestamp('20111230'), Timestamp('20120101'), Timestamp('20120103')])
        dt1.iloc[2] = np.nan
        dt2 = Series([Timestamp('20111231'), Timestamp('20120102'), Timestamp('20120104')])
        dt1 - dt2
        dt2 - dt1
        dt1 + td1
        td1 + dt1
        dt1 - td1
        td1 + dt1
        dt1 + td1

    def test_dt64ser_sub_datetime_dtype(self, unit):
        ts = Timestamp(datetime(1993, 1, 7, 13, 30, 0))
        dt = datetime(1993, 6, 22, 13, 30)
        ser = Series([ts], dtype=f'M8[{unit}]')
        result = ser - dt
        exp_unit = tm.get_finest_unit(unit, 'us')
        assert result.dtype == f'timedelta64[{exp_unit}]'

    @pytest.mark.parametrize('left, right, op_fail', [[[Timestamp('20111230'), Timestamp('20120101'), NaT], [Timestamp('20111231'), Timestamp('20120102'), Timestamp('20120104')], ['__sub__', '__rsub__']], [[Timestamp('20111230'), Timestamp('20120101'), NaT], [timedelta(minutes=5, seconds=3), timedelta(minutes=5, seconds=3), NaT], ['__add__', '__radd__', '__sub__']], [[Timestamp('20111230', tz='US/Eastern'), Timestamp('20111230', tz='US/Eastern'), NaT], [timedelta(minutes=5, seconds=3), NaT, timedelta(minutes=5, seconds=3)], ['__add__', '__radd__', '__sub__']]])
    def test_operators_datetimelike_invalid(self, left, right, op_fail, all_arithmetic_operators):
        op_str = all_arithmetic_operators
        arg1 = Series(left)
        arg2 = Series(right)
        op = getattr(arg1, op_str, None)
        if op_str not in op_fail:
            with pytest.raises(TypeError, match='operate|[cC]annot|unsupported operand'):
                op(arg2)
        else:
            op(arg2)

    def test_sub_single_tz(self, unit):
        s1 = Series([Timestamp('2016-02-10', tz='America/Sao_Paulo')]).dt.as_unit(unit)
        s2 = Series([Timestamp('2016-02-08', tz='America/Sao_Paulo')]).dt.as_unit(unit)
        result = s1 - s2
        expected = Series([Timedelta('2days')]).dt.as_unit(unit)
        tm.assert_series_equal(result, expected)
        result = s2 - s1
        expected = Series([Timedelta('-2days')]).dt.as_unit(unit)
        tm.assert_series_equal(result, expected)

    def test_dt64tz_series_sub_dtitz(self):
        dti = date_range('1999-09-30', periods=10, tz='US/Pacific')
        ser = Series(dti)
        expected = Series(TimedeltaIndex(['0days'] * 10))
        res = dti - ser
        tm.assert_series_equal(res, expected)
        res = ser - dti
        tm.assert_series_equal(res, expected)

    def test_sub_datetime_compat(self, unit):
        ser = Series([datetime(2016, 8, 23, 12, tzinfo=pytz.utc), NaT]).dt.as_unit(unit)
        dt = datetime(2016, 8, 22, 12, tzinfo=pytz.utc)
        exp_unit = tm.get_finest_unit(unit, 'us')
        exp = Series([Timedelta('1 days'), NaT]).dt.as_unit(exp_unit)
        result = ser - dt
        tm.assert_series_equal(result, exp)
        result2 = ser - Timestamp(dt)
        tm.assert_series_equal(result2, exp)

    def test_dt64_series_add_mixed_tick_DateOffset(self):
        s = Series([Timestamp('20130101 9:01'), Timestamp('20130101 9:02')])
        result = s + pd.offsets.Milli(5)
        result2 = pd.offsets.Milli(5) + s
        expected = Series([Timestamp('20130101 9:01:00.005'), Timestamp('20130101 9:02:00.005')])
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(result2, expected)
        result = s + pd.offsets.Minute(5) + pd.offsets.Milli(5)
        expected = Series([Timestamp('20130101 9:06:00.005'), Timestamp('20130101 9:07:00.005')])
        tm.assert_series_equal(result, expected)

    def test_datetime64_ops_nat(self, unit):
        datetime_series = Series([NaT, Timestamp('19900315')]).dt.as_unit(unit)
        nat_series_dtype_timestamp = Series([NaT, NaT], dtype=f'datetime64[{unit}]')
        single_nat_dtype_datetime = Series([NaT], dtype=f'datetime64[{unit}]')
        tm.assert_series_equal(-NaT + datetime_series, nat_series_dtype_timestamp)
        msg = "bad operand type for unary -: 'DatetimeArray'"
        with pytest.raises(TypeError, match=msg):
            -single_nat_dtype_datetime + datetime_series
        tm.assert_series_equal(-NaT + nat_series_dtype_timestamp, nat_series_dtype_timestamp)
        with pytest.raises(TypeError, match=msg):
            -single_nat_dtype_datetime + nat_series_dtype_timestamp
        tm.assert_series_equal(nat_series_dtype_timestamp + NaT, nat_series_dtype_timestamp)
        tm.assert_series_equal(NaT + nat_series_dtype_timestamp, nat_series_dtype_timestamp)
        tm.assert_series_equal(nat_series_dtype_timestamp + NaT, nat_series_dtype_timestamp)
        tm.assert_series_equal(NaT + nat_series_dtype_timestamp, nat_series_dtype_timestamp)

    def test_operators_datetimelike_with_timezones(self):
        tz = 'US/Eastern'
        dt1 = Series(date_range('2000-01-01 09:00:00', periods=5, tz=tz), name='foo')
        dt2 = dt1.copy()
        dt2.iloc[2] = np.nan
        td1 = Series(pd.timedelta_range('1 days 1 min', periods=5, freq='h'))
        td2 = td1.copy()
        td2.iloc[1] = np.nan
        assert td2._values.freq is None
        result = dt1 + td1[0]
        exp = (dt1.dt.tz_localize(None) + td1[0]).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        result = dt2 + td2[0]
        exp = (dt2.dt.tz_localize(None) + td2[0]).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        result = td1[0] + dt1
        exp = (dt1.dt.tz_localize(None) + td1[0]).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        result = td2[0] + dt2
        exp = (dt2.dt.tz_localize(None) + td2[0]).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        result = dt1 - td1[0]
        exp = (dt1.dt.tz_localize(None) - td1[0]).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        msg = '(bad|unsupported) operand type for unary'
        with pytest.raises(TypeError, match=msg):
            td1[0] - dt1
        result = dt2 - td2[0]
        exp = (dt2.dt.tz_localize(None) - td2[0]).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        with pytest.raises(TypeError, match=msg):
            td2[0] - dt2
        result = dt1 + td1
        exp = (dt1.dt.tz_localize(None) + td1).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        result = dt2 + td2
        exp = (dt2.dt.tz_localize(None) + td2).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        result = dt1 - td1
        exp = (dt1.dt.tz_localize(None) - td1).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        result = dt2 - td2
        exp = (dt2.dt.tz_localize(None) - td2).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        msg = 'cannot (add|subtract)'
        with pytest.raises(TypeError, match=msg):
            td1 - dt1
        with pytest.raises(TypeError, match=msg):
            td2 - dt2