from __future__ import annotations
from collections import abc
from datetime import timedelta
from decimal import Decimal
import operator
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.tests.arithmetic.common import (
class TestNumericArithmeticUnsorted:

    @pytest.mark.parametrize('op', [operator.add, operator.sub, operator.mul, operator.floordiv, operator.truediv])
    @pytest.mark.parametrize('idx1', [RangeIndex(0, 10, 1), RangeIndex(0, 20, 2), RangeIndex(-10, 10, 2), RangeIndex(5, -5, -1)])
    @pytest.mark.parametrize('idx2', [RangeIndex(0, 10, 1), RangeIndex(0, 20, 2), RangeIndex(-10, 10, 2), RangeIndex(5, -5, -1)])
    def test_binops_index(self, op, idx1, idx2):
        idx1 = idx1._rename('foo')
        idx2 = idx2._rename('bar')
        result = op(idx1, idx2)
        expected = op(Index(idx1.to_numpy()), Index(idx2.to_numpy()))
        tm.assert_index_equal(result, expected, exact='equiv')

    @pytest.mark.parametrize('op', [operator.add, operator.sub, operator.mul, operator.floordiv, operator.truediv])
    @pytest.mark.parametrize('idx', [RangeIndex(0, 10, 1), RangeIndex(0, 20, 2), RangeIndex(-10, 10, 2), RangeIndex(5, -5, -1)])
    @pytest.mark.parametrize('scalar', [-1, 1, 2])
    def test_binops_index_scalar(self, op, idx, scalar):
        result = op(idx, scalar)
        expected = op(Index(idx.to_numpy()), scalar)
        tm.assert_index_equal(result, expected, exact='equiv')

    @pytest.mark.parametrize('idx1', [RangeIndex(0, 10, 1), RangeIndex(0, 20, 2)])
    @pytest.mark.parametrize('idx2', [RangeIndex(0, 10, 1), RangeIndex(0, 20, 2)])
    def test_binops_index_pow(self, idx1, idx2):
        idx1 = idx1._rename('foo')
        idx2 = idx2._rename('bar')
        result = pow(idx1, idx2)
        expected = pow(Index(idx1.to_numpy()), Index(idx2.to_numpy()))
        tm.assert_index_equal(result, expected, exact='equiv')

    @pytest.mark.parametrize('idx', [RangeIndex(0, 10, 1), RangeIndex(0, 20, 2)])
    @pytest.mark.parametrize('scalar', [1, 2])
    def test_binops_index_scalar_pow(self, idx, scalar):
        result = pow(idx, scalar)
        expected = pow(Index(idx.to_numpy()), scalar)
        tm.assert_index_equal(result, expected, exact='equiv')

    @pytest.mark.parametrize('op', [operator.add, operator.sub, operator.mul, operator.floordiv, operator.truediv, operator.pow, operator.mod])
    def test_arithmetic_with_frame_or_series(self, op):
        index = RangeIndex(5)
        other = Series(np.random.default_rng(2).standard_normal(5))
        expected = op(Series(index), other)
        result = op(index, other)
        tm.assert_series_equal(result, expected)
        other = pd.DataFrame(np.random.default_rng(2).standard_normal((2, 5)))
        expected = op(pd.DataFrame([index, index]), other)
        result = op(index, other)
        tm.assert_frame_equal(result, expected)

    def test_numeric_compat2(self):
        idx = RangeIndex(0, 10, 2)
        result = idx * 2
        expected = RangeIndex(0, 20, 4)
        tm.assert_index_equal(result, expected, exact=True)
        result = idx + 2
        expected = RangeIndex(2, 12, 2)
        tm.assert_index_equal(result, expected, exact=True)
        result = idx - 2
        expected = RangeIndex(-2, 8, 2)
        tm.assert_index_equal(result, expected, exact=True)
        result = idx / 2
        expected = RangeIndex(0, 5, 1).astype('float64')
        tm.assert_index_equal(result, expected, exact=True)
        result = idx / 4
        expected = RangeIndex(0, 10, 2) / 4
        tm.assert_index_equal(result, expected, exact=True)
        result = idx // 1
        expected = idx
        tm.assert_index_equal(result, expected, exact=True)
        result = idx * idx
        expected = Index(idx.values * idx.values)
        tm.assert_index_equal(result, expected, exact=True)
        idx = RangeIndex(0, 1000, 2)
        result = idx ** 2
        expected = Index(idx._values) ** 2
        tm.assert_index_equal(Index(result.values), expected, exact=True)

    @pytest.mark.parametrize('idx, div, expected', [(RangeIndex(0, 1000, 2), 2, RangeIndex(0, 500, 1)), (RangeIndex(-99, -201, -3), -3, RangeIndex(33, 67, 1)), (RangeIndex(0, 1000, 1), 2, Index(RangeIndex(0, 1000, 1)._values) // 2), (RangeIndex(0, 100, 1), 2.0, Index(RangeIndex(0, 100, 1)._values) // 2.0), (RangeIndex(0), 50, RangeIndex(0)), (RangeIndex(2, 4, 2), 3, RangeIndex(0, 1, 1)), (RangeIndex(-5, -10, -6), 4, RangeIndex(-2, -1, 1)), (RangeIndex(-100, -200, 3), 2, RangeIndex(0))])
    def test_numeric_compat2_floordiv(self, idx, div, expected):
        tm.assert_index_equal(idx // div, expected, exact=True)

    @pytest.mark.parametrize('dtype', [np.int64, np.float64])
    @pytest.mark.parametrize('delta', [1, 0, -1])
    def test_addsub_arithmetic(self, dtype, delta):
        delta = dtype(delta)
        index = Index([10, 11, 12], dtype=dtype)
        result = index + delta
        expected = Index(index.values + delta, dtype=dtype)
        tm.assert_index_equal(result, expected)
        result = index - delta
        expected = Index(index.values - delta, dtype=dtype)
        tm.assert_index_equal(result, expected)
        tm.assert_index_equal(index + index, 2 * index)
        tm.assert_index_equal(index - index, 0 * index)
        assert not (index - index).empty

    def test_pow_nan_with_zero(self, box_with_array):
        left = Index([np.nan, np.nan, np.nan])
        right = Index([0, 0, 0])
        expected = Index([1.0, 1.0, 1.0])
        left = tm.box_expected(left, box_with_array)
        right = tm.box_expected(right, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        result = left ** right
        tm.assert_equal(result, expected)