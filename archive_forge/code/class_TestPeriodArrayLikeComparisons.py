import operator
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import TimedeltaArray
from pandas.tests.arithmetic.common import (
class TestPeriodArrayLikeComparisons:

    @pytest.mark.parametrize('other', ['2017', Period('2017', freq='D')])
    def test_eq_scalar(self, other, box_with_array):
        idx = PeriodIndex(['2017', '2017', '2018'], freq='D')
        idx = tm.box_expected(idx, box_with_array)
        xbox = get_upcast_box(idx, other, True)
        expected = np.array([True, True, False])
        expected = tm.box_expected(expected, xbox)
        result = idx == other
        tm.assert_equal(result, expected)

    def test_compare_zerodim(self, box_with_array):
        pi = period_range('2000', periods=4)
        other = np.array(pi.to_numpy()[0])
        pi = tm.box_expected(pi, box_with_array)
        xbox = get_upcast_box(pi, other, True)
        result = pi <= other
        expected = np.array([True, False, False, False])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('scalar', ['foo', Timestamp('2021-01-01'), Timedelta(days=4), 9, 9.5, 2000, False, None])
    def test_compare_invalid_scalar(self, box_with_array, scalar):
        pi = period_range('2000', periods=4)
        parr = tm.box_expected(pi, box_with_array)
        assert_invalid_comparison(parr, scalar, box_with_array)

    @pytest.mark.parametrize('other', [pd.date_range('2000', periods=4).array, pd.timedelta_range('1D', periods=4).array, np.arange(4), np.arange(4).astype(np.float64), list(range(4)), [2000, 2001, 2002, 2003], np.arange(2000, 2004), np.arange(2000, 2004).astype(object), pd.Index([2000, 2001, 2002, 2003])])
    def test_compare_invalid_listlike(self, box_with_array, other):
        pi = period_range('2000', periods=4)
        parr = tm.box_expected(pi, box_with_array)
        assert_invalid_comparison(parr, other, box_with_array)

    @pytest.mark.parametrize('other_box', [list, np.array, lambda x: x.astype(object)])
    def test_compare_object_dtype(self, box_with_array, other_box):
        pi = period_range('2000', periods=5)
        parr = tm.box_expected(pi, box_with_array)
        other = other_box(pi)
        xbox = get_upcast_box(parr, other, True)
        expected = np.array([True, True, True, True, True])
        expected = tm.box_expected(expected, xbox)
        result = parr == other
        tm.assert_equal(result, expected)
        result = parr <= other
        tm.assert_equal(result, expected)
        result = parr >= other
        tm.assert_equal(result, expected)
        result = parr != other
        tm.assert_equal(result, ~expected)
        result = parr < other
        tm.assert_equal(result, ~expected)
        result = parr > other
        tm.assert_equal(result, ~expected)
        other = other_box(pi[::-1])
        expected = np.array([False, False, True, False, False])
        expected = tm.box_expected(expected, xbox)
        result = parr == other
        tm.assert_equal(result, expected)
        expected = np.array([True, True, True, False, False])
        expected = tm.box_expected(expected, xbox)
        result = parr <= other
        tm.assert_equal(result, expected)
        expected = np.array([False, False, True, True, True])
        expected = tm.box_expected(expected, xbox)
        result = parr >= other
        tm.assert_equal(result, expected)
        expected = np.array([True, True, False, True, True])
        expected = tm.box_expected(expected, xbox)
        result = parr != other
        tm.assert_equal(result, expected)
        expected = np.array([True, True, False, False, False])
        expected = tm.box_expected(expected, xbox)
        result = parr < other
        tm.assert_equal(result, expected)
        expected = np.array([False, False, False, True, True])
        expected = tm.box_expected(expected, xbox)
        result = parr > other
        tm.assert_equal(result, expected)