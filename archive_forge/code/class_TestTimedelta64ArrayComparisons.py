from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
class TestTimedelta64ArrayComparisons:

    @pytest.mark.parametrize('dtype', [None, object])
    def test_comp_nat(self, dtype):
        left = TimedeltaIndex([Timedelta('1 days'), NaT, Timedelta('3 days')])
        right = TimedeltaIndex([NaT, NaT, Timedelta('3 days')])
        lhs, rhs = (left, right)
        if dtype is object:
            lhs, rhs = (left.astype(object), right.astype(object))
        result = rhs == lhs
        expected = np.array([False, False, True])
        tm.assert_numpy_array_equal(result, expected)
        result = rhs != lhs
        expected = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)
        expected = np.array([False, False, False])
        tm.assert_numpy_array_equal(lhs == NaT, expected)
        tm.assert_numpy_array_equal(NaT == rhs, expected)
        expected = np.array([True, True, True])
        tm.assert_numpy_array_equal(lhs != NaT, expected)
        tm.assert_numpy_array_equal(NaT != lhs, expected)
        expected = np.array([False, False, False])
        tm.assert_numpy_array_equal(lhs < NaT, expected)
        tm.assert_numpy_array_equal(NaT > lhs, expected)

    @pytest.mark.parametrize('idx2', [TimedeltaIndex(['2 day', '2 day', NaT, NaT, '1 day 00:00:02', '5 days 00:00:03']), np.array([np.timedelta64(2, 'D'), np.timedelta64(2, 'D'), np.timedelta64('nat'), np.timedelta64('nat'), np.timedelta64(1, 'D') + np.timedelta64(2, 's'), np.timedelta64(5, 'D') + np.timedelta64(3, 's')])])
    def test_comparisons_nat(self, idx2):
        idx1 = TimedeltaIndex(['1 day', NaT, '1 day 00:00:01', NaT, '1 day 00:00:01', '5 day 00:00:03'])
        result = idx1 < idx2
        expected = np.array([True, False, False, False, True, False])
        tm.assert_numpy_array_equal(result, expected)
        result = idx2 > idx1
        expected = np.array([True, False, False, False, True, False])
        tm.assert_numpy_array_equal(result, expected)
        result = idx1 <= idx2
        expected = np.array([True, False, False, False, True, True])
        tm.assert_numpy_array_equal(result, expected)
        result = idx2 >= idx1
        expected = np.array([True, False, False, False, True, True])
        tm.assert_numpy_array_equal(result, expected)
        result = idx1 == idx2
        expected = np.array([False, False, False, False, False, True])
        tm.assert_numpy_array_equal(result, expected)
        result = idx1 != idx2
        expected = np.array([True, True, True, True, True, False])
        tm.assert_numpy_array_equal(result, expected)

    def test_comparisons_coverage(self):
        rng = timedelta_range('1 days', periods=10)
        result = rng < rng[3]
        expected = np.array([True, True, True] + [False] * 7)
        tm.assert_numpy_array_equal(result, expected)
        result = rng == list(rng)
        exp = rng == rng
        tm.assert_numpy_array_equal(result, exp)