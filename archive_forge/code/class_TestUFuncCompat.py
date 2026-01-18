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
class TestUFuncCompat:

    @pytest.mark.parametrize('holder', [Index, RangeIndex, Series])
    @pytest.mark.parametrize('dtype', [np.int64, np.uint64, np.float64])
    def test_ufunc_compat(self, holder, dtype):
        box = Series if holder is Series else Index
        if holder is RangeIndex:
            if dtype != np.int64:
                pytest.skip(f'dtype {dtype} not relevant for RangeIndex')
            idx = RangeIndex(0, 5, name='foo')
        else:
            idx = holder(np.arange(5, dtype=dtype), name='foo')
        result = np.sin(idx)
        expected = box(np.sin(np.arange(5, dtype=dtype)), name='foo')
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('holder', [Index, Series])
    @pytest.mark.parametrize('dtype', [np.int64, np.uint64, np.float64])
    def test_ufunc_coercions(self, holder, dtype):
        idx = holder([1, 2, 3, 4, 5], dtype=dtype, name='x')
        box = Series if holder is Series else Index
        result = np.sqrt(idx)
        assert result.dtype == 'f8' and isinstance(result, box)
        exp = Index(np.sqrt(np.array([1, 2, 3, 4, 5], dtype=np.float64)), name='x')
        exp = tm.box_expected(exp, box)
        tm.assert_equal(result, exp)
        result = np.divide(idx, 2.0)
        assert result.dtype == 'f8' and isinstance(result, box)
        exp = Index([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64, name='x')
        exp = tm.box_expected(exp, box)
        tm.assert_equal(result, exp)
        result = idx + 2.0
        assert result.dtype == 'f8' and isinstance(result, box)
        exp = Index([3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float64, name='x')
        exp = tm.box_expected(exp, box)
        tm.assert_equal(result, exp)
        result = idx - 2.0
        assert result.dtype == 'f8' and isinstance(result, box)
        exp = Index([-1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float64, name='x')
        exp = tm.box_expected(exp, box)
        tm.assert_equal(result, exp)
        result = idx * 1.0
        assert result.dtype == 'f8' and isinstance(result, box)
        exp = Index([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64, name='x')
        exp = tm.box_expected(exp, box)
        tm.assert_equal(result, exp)
        result = idx / 2.0
        assert result.dtype == 'f8' and isinstance(result, box)
        exp = Index([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64, name='x')
        exp = tm.box_expected(exp, box)
        tm.assert_equal(result, exp)

    @pytest.mark.parametrize('holder', [Index, Series])
    @pytest.mark.parametrize('dtype', [np.int64, np.uint64, np.float64])
    def test_ufunc_multiple_return_values(self, holder, dtype):
        obj = holder([1, 2, 3], dtype=dtype, name='x')
        box = Series if holder is Series else Index
        result = np.modf(obj)
        assert isinstance(result, tuple)
        exp1 = Index([0.0, 0.0, 0.0], dtype=np.float64, name='x')
        exp2 = Index([1.0, 2.0, 3.0], dtype=np.float64, name='x')
        tm.assert_equal(result[0], tm.box_expected(exp1, box))
        tm.assert_equal(result[1], tm.box_expected(exp2, box))

    def test_ufunc_at(self):
        s = Series([0, 1, 2], index=[1, 2, 3], name='x')
        np.add.at(s, [0, 2], 10)
        expected = Series([10, 1, 12], index=[1, 2, 3], name='x')
        tm.assert_series_equal(s, expected)