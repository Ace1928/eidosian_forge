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
class TestDatetime64SeriesComparison:

    @pytest.mark.parametrize('pair', [([Timestamp('2011-01-01'), NaT, Timestamp('2011-01-03')], [NaT, NaT, Timestamp('2011-01-03')]), ([Timedelta('1 days'), NaT, Timedelta('3 days')], [NaT, NaT, Timedelta('3 days')]), ([Period('2011-01', freq='M'), NaT, Period('2011-03', freq='M')], [NaT, NaT, Period('2011-03', freq='M')])])
    @pytest.mark.parametrize('reverse', [True, False])
    @pytest.mark.parametrize('dtype', [None, object])
    @pytest.mark.parametrize('op, expected', [(operator.eq, Series([False, False, True])), (operator.ne, Series([True, True, False])), (operator.lt, Series([False, False, False])), (operator.gt, Series([False, False, False])), (operator.ge, Series([False, False, True])), (operator.le, Series([False, False, True]))])
    def test_nat_comparisons(self, dtype, index_or_series, reverse, pair, op, expected):
        box = index_or_series
        lhs, rhs = pair
        if reverse:
            lhs, rhs = (rhs, lhs)
        left = Series(lhs, dtype=dtype)
        right = box(rhs, dtype=dtype)
        result = op(left, right)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('data', [[Timestamp('2011-01-01'), NaT, Timestamp('2011-01-03')], [Timedelta('1 days'), NaT, Timedelta('3 days')], [Period('2011-01', freq='M'), NaT, Period('2011-03', freq='M')]])
    @pytest.mark.parametrize('dtype', [None, object])
    def test_nat_comparisons_scalar(self, dtype, data, box_with_array):
        box = box_with_array
        left = Series(data, dtype=dtype)
        left = tm.box_expected(left, box)
        xbox = get_upcast_box(left, NaT, True)
        expected = [False, False, False]
        expected = tm.box_expected(expected, xbox)
        if box is pd.array and dtype is object:
            expected = pd.array(expected, dtype='bool')
        tm.assert_equal(left == NaT, expected)
        tm.assert_equal(NaT == left, expected)
        expected = [True, True, True]
        expected = tm.box_expected(expected, xbox)
        if box is pd.array and dtype is object:
            expected = pd.array(expected, dtype='bool')
        tm.assert_equal(left != NaT, expected)
        tm.assert_equal(NaT != left, expected)
        expected = [False, False, False]
        expected = tm.box_expected(expected, xbox)
        if box is pd.array and dtype is object:
            expected = pd.array(expected, dtype='bool')
        tm.assert_equal(left < NaT, expected)
        tm.assert_equal(NaT > left, expected)
        tm.assert_equal(left <= NaT, expected)
        tm.assert_equal(NaT >= left, expected)
        tm.assert_equal(left > NaT, expected)
        tm.assert_equal(NaT < left, expected)
        tm.assert_equal(left >= NaT, expected)
        tm.assert_equal(NaT <= left, expected)

    @pytest.mark.parametrize('val', [datetime(2000, 1, 4), datetime(2000, 1, 5)])
    def test_series_comparison_scalars(self, val):
        series = Series(date_range('1/1/2000', periods=10))
        result = series > val
        expected = Series([x > val for x in series])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('left,right', [('lt', 'gt'), ('le', 'ge'), ('eq', 'eq'), ('ne', 'ne')])
    def test_timestamp_compare_series(self, left, right):
        ser = Series(date_range('20010101', periods=10), name='dates')
        s_nat = ser.copy(deep=True)
        ser[0] = Timestamp('nat')
        ser[3] = Timestamp('nat')
        left_f = getattr(operator, left)
        right_f = getattr(operator, right)
        expected = left_f(ser, Timestamp('20010109'))
        result = right_f(Timestamp('20010109'), ser)
        tm.assert_series_equal(result, expected)
        expected = left_f(ser, Timestamp('nat'))
        result = right_f(Timestamp('nat'), ser)
        tm.assert_series_equal(result, expected)
        expected = left_f(s_nat, Timestamp('20010109'))
        result = right_f(Timestamp('20010109'), s_nat)
        tm.assert_series_equal(result, expected)
        expected = left_f(s_nat, NaT)
        result = right_f(NaT, s_nat)
        tm.assert_series_equal(result, expected)

    def test_dt64arr_timestamp_equality(self, box_with_array):
        box = box_with_array
        ser = Series([Timestamp('2000-01-29 01:59:00'), Timestamp('2000-01-30'), NaT])
        ser = tm.box_expected(ser, box)
        xbox = get_upcast_box(ser, ser, True)
        result = ser != ser
        expected = tm.box_expected([False, False, True], xbox)
        tm.assert_equal(result, expected)
        if box is pd.DataFrame:
            with pytest.raises(ValueError, match='not aligned'):
                ser != ser[0]
        else:
            result = ser != ser[0]
            expected = tm.box_expected([False, True, True], xbox)
            tm.assert_equal(result, expected)
        if box is pd.DataFrame:
            with pytest.raises(ValueError, match='not aligned'):
                ser != ser[2]
        else:
            result = ser != ser[2]
            expected = tm.box_expected([True, True, True], xbox)
            tm.assert_equal(result, expected)
        result = ser == ser
        expected = tm.box_expected([True, True, False], xbox)
        tm.assert_equal(result, expected)
        if box is pd.DataFrame:
            with pytest.raises(ValueError, match='not aligned'):
                ser == ser[0]
        else:
            result = ser == ser[0]
            expected = tm.box_expected([True, False, False], xbox)
            tm.assert_equal(result, expected)
        if box is pd.DataFrame:
            with pytest.raises(ValueError, match='not aligned'):
                ser == ser[2]
        else:
            result = ser == ser[2]
            expected = tm.box_expected([False, False, False], xbox)
            tm.assert_equal(result, expected)

    @pytest.mark.parametrize('datetimelike', [Timestamp('20130101'), datetime(2013, 1, 1), np.datetime64('2013-01-01T00:00', 'ns')])
    @pytest.mark.parametrize('op,expected', [(operator.lt, [True, False, False, False]), (operator.le, [True, True, False, False]), (operator.eq, [False, True, False, False]), (operator.gt, [False, False, False, True])])
    def test_dt64_compare_datetime_scalar(self, datetimelike, op, expected):
        ser = Series([Timestamp('20120101'), Timestamp('20130101'), np.nan, Timestamp('20130103')], name='A')
        result = op(ser, datetimelike)
        expected = Series(expected, name='A')
        tm.assert_series_equal(result, expected)