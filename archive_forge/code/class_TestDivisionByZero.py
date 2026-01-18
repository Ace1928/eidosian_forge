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
class TestDivisionByZero:

    def test_div_zero(self, zero, numeric_idx):
        idx = numeric_idx
        expected = Index([np.nan, np.inf, np.inf, np.inf, np.inf], dtype=np.float64)
        expected2 = adjust_negative_zero(zero, expected)
        result = idx / zero
        tm.assert_index_equal(result, expected2)
        ser_compat = Series(idx).astype('i8') / np.array(zero).astype('i8')
        tm.assert_series_equal(ser_compat, Series(expected))

    def test_floordiv_zero(self, zero, numeric_idx):
        idx = numeric_idx
        expected = Index([np.nan, np.inf, np.inf, np.inf, np.inf], dtype=np.float64)
        expected2 = adjust_negative_zero(zero, expected)
        result = idx // zero
        tm.assert_index_equal(result, expected2)
        ser_compat = Series(idx).astype('i8') // np.array(zero).astype('i8')
        tm.assert_series_equal(ser_compat, Series(expected))

    def test_mod_zero(self, zero, numeric_idx):
        idx = numeric_idx
        expected = Index([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
        result = idx % zero
        tm.assert_index_equal(result, expected)
        ser_compat = Series(idx).astype('i8') % np.array(zero).astype('i8')
        tm.assert_series_equal(ser_compat, Series(result))

    def test_divmod_zero(self, zero, numeric_idx):
        idx = numeric_idx
        exleft = Index([np.nan, np.inf, np.inf, np.inf, np.inf], dtype=np.float64)
        exright = Index([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
        exleft = adjust_negative_zero(zero, exleft)
        result = divmod(idx, zero)
        tm.assert_index_equal(result[0], exleft)
        tm.assert_index_equal(result[1], exright)

    @pytest.mark.parametrize('op', [operator.truediv, operator.floordiv])
    def test_div_negative_zero(self, zero, numeric_idx, op):
        if numeric_idx.dtype == np.uint64:
            pytest.skip(f'Div by negative 0 not relevant for {numeric_idx.dtype}')
        idx = numeric_idx - 3
        expected = Index([-np.inf, -np.inf, -np.inf, np.nan, np.inf], dtype=np.float64)
        expected = adjust_negative_zero(zero, expected)
        result = op(idx, zero)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('dtype1', [np.int64, np.float64, np.uint64])
    def test_ser_div_ser(self, switch_numexpr_min_elements, dtype1, any_real_numpy_dtype):
        dtype2 = any_real_numpy_dtype
        first = Series([3, 4, 5, 8], name='first').astype(dtype1)
        second = Series([0, 0, 0, 3], name='second').astype(dtype2)
        with np.errstate(all='ignore'):
            expected = Series(first.values.astype(np.float64) / second.values, dtype='float64', name=None)
        expected.iloc[0:3] = np.inf
        if first.dtype == 'int64' and second.dtype == 'float32':
            if expr.USE_NUMEXPR and switch_numexpr_min_elements == 0:
                expected = expected.astype('float32')
        result = first / second
        tm.assert_series_equal(result, expected)
        assert not result.equals(second / first)

    @pytest.mark.parametrize('dtype1', [np.int64, np.float64, np.uint64])
    def test_ser_divmod_zero(self, dtype1, any_real_numpy_dtype):
        dtype2 = any_real_numpy_dtype
        left = Series([1, 1]).astype(dtype1)
        right = Series([0, 2]).astype(dtype2)
        expected = (left // right, left % right)
        expected = list(expected)
        expected[0] = expected[0].astype(np.float64)
        expected[0][0] = np.inf
        result = divmod(left, right)
        tm.assert_series_equal(result[0], expected[0])
        tm.assert_series_equal(result[1], expected[1])
        result = divmod(left.values, right)
        tm.assert_series_equal(result[0], expected[0])
        tm.assert_series_equal(result[1], expected[1])

    def test_ser_divmod_inf(self):
        left = Series([np.inf, 1.0])
        right = Series([np.inf, 2.0])
        expected = (left // right, left % right)
        result = divmod(left, right)
        tm.assert_series_equal(result[0], expected[0])
        tm.assert_series_equal(result[1], expected[1])
        result = divmod(left.values, right)
        tm.assert_series_equal(result[0], expected[0])
        tm.assert_series_equal(result[1], expected[1])

    def test_rdiv_zero_compat(self):
        zero_array = np.array([0] * 5)
        data = np.random.default_rng(2).standard_normal(5)
        expected = Series([0.0] * 5)
        result = zero_array / Series(data)
        tm.assert_series_equal(result, expected)
        result = Series(zero_array) / data
        tm.assert_series_equal(result, expected)
        result = Series(zero_array) / Series(data)
        tm.assert_series_equal(result, expected)

    def test_div_zero_inf_signs(self):
        ser = Series([-1, 0, 1], name='first')
        expected = Series([-np.inf, np.nan, np.inf], name='first')
        result = ser / 0
        tm.assert_series_equal(result, expected)

    def test_rdiv_zero(self):
        ser = Series([-1, 0, 1], name='first')
        expected = Series([0.0, np.nan, 0.0], name='first')
        result = 0 / ser
        tm.assert_series_equal(result, expected)

    def test_floordiv_div(self):
        ser = Series([-1, 0, 1], name='first')
        result = ser // 0
        expected = Series([-np.inf, np.nan, np.inf], name='first')
        tm.assert_series_equal(result, expected)

    def test_df_div_zero_df(self):
        df = pd.DataFrame({'first': [3, 4, 5, 8], 'second': [0, 0, 0, 3]})
        result = df / df
        first = Series([1.0, 1.0, 1.0, 1.0])
        second = Series([np.nan, np.nan, np.nan, 1])
        expected = pd.DataFrame({'first': first, 'second': second})
        tm.assert_frame_equal(result, expected)

    def test_df_div_zero_array(self):
        df = pd.DataFrame({'first': [3, 4, 5, 8], 'second': [0, 0, 0, 3]})
        first = Series([1.0, 1.0, 1.0, 1.0])
        second = Series([np.nan, np.nan, np.nan, 1])
        expected = pd.DataFrame({'first': first, 'second': second})
        with np.errstate(all='ignore'):
            arr = df.values.astype('float') / df.values
        result = pd.DataFrame(arr, index=df.index, columns=df.columns)
        tm.assert_frame_equal(result, expected)

    def test_df_div_zero_int(self):
        df = pd.DataFrame({'first': [3, 4, 5, 8], 'second': [0, 0, 0, 3]})
        result = df / 0
        expected = pd.DataFrame(np.inf, index=df.index, columns=df.columns)
        expected.iloc[0:3, 1] = np.nan
        tm.assert_frame_equal(result, expected)
        with np.errstate(all='ignore'):
            arr = df.values.astype('float64') / 0
        result2 = pd.DataFrame(arr, index=df.index, columns=df.columns)
        tm.assert_frame_equal(result2, expected)

    def test_df_div_zero_series_does_not_commute(self):
        df = pd.DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
        ser = df[0]
        res = ser / df
        res2 = df / ser
        assert not res.fillna(0).equals(res2.fillna(0))

    def test_df_mod_zero_df(self, using_array_manager):
        df = pd.DataFrame({'first': [3, 4, 5, 8], 'second': [0, 0, 0, 3]})
        first = Series([0, 0, 0, 0])
        if not using_array_manager:
            first = first.astype('float64')
        second = Series([np.nan, np.nan, np.nan, 0])
        expected = pd.DataFrame({'first': first, 'second': second})
        result = df % df
        tm.assert_frame_equal(result, expected)
        df = pd.DataFrame({'first': [3, 4, 5, 8], 'second': [0, 0, 0, 3]}, copy=False)
        first = Series([0, 0, 0, 0], dtype='int64')
        second = Series([np.nan, np.nan, np.nan, 0])
        expected = pd.DataFrame({'first': first, 'second': second})
        result = df % df
        tm.assert_frame_equal(result, expected)

    def test_df_mod_zero_array(self):
        df = pd.DataFrame({'first': [3, 4, 5, 8], 'second': [0, 0, 0, 3]})
        first = Series([0, 0, 0, 0], dtype='float64')
        second = Series([np.nan, np.nan, np.nan, 0])
        expected = pd.DataFrame({'first': first, 'second': second})
        with np.errstate(all='ignore'):
            arr = df.values % df.values
        result2 = pd.DataFrame(arr, index=df.index, columns=df.columns, dtype='float64')
        result2.iloc[0:3, 1] = np.nan
        tm.assert_frame_equal(result2, expected)

    def test_df_mod_zero_int(self):
        df = pd.DataFrame({'first': [3, 4, 5, 8], 'second': [0, 0, 0, 3]})
        result = df % 0
        expected = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
        tm.assert_frame_equal(result, expected)
        with np.errstate(all='ignore'):
            arr = df.values.astype('float64') % 0
        result2 = pd.DataFrame(arr, index=df.index, columns=df.columns)
        tm.assert_frame_equal(result2, expected)

    def test_df_mod_zero_series_does_not_commute(self):
        df = pd.DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
        ser = df[0]
        res = ser % df
        res2 = df % ser
        assert not res.fillna(0).equals(res2.fillna(0))