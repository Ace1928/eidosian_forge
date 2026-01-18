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
class TestMultiplicationDivision:

    def test_divide_decimal(self, box_with_array):
        box = box_with_array
        ser = Series([Decimal(10)])
        expected = Series([Decimal(5)])
        ser = tm.box_expected(ser, box)
        expected = tm.box_expected(expected, box)
        result = ser / Decimal(2)
        tm.assert_equal(result, expected)
        result = ser // Decimal(2)
        tm.assert_equal(result, expected)

    def test_div_equiv_binop(self):
        first = Series([1, 0], name='first')
        second = Series([-0.01, -0.02], name='second')
        expected = Series([-0.01, -np.inf])
        result = second.div(first)
        tm.assert_series_equal(result, expected, check_names=False)
        result = second / first
        tm.assert_series_equal(result, expected)

    def test_div_int(self, numeric_idx):
        idx = numeric_idx
        result = idx / 1
        expected = idx.astype('float64')
        tm.assert_index_equal(result, expected)
        result = idx / 2
        expected = Index(idx.values / 2)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('op', [operator.mul, ops.rmul, operator.floordiv])
    def test_mul_int_identity(self, op, numeric_idx, box_with_array):
        idx = numeric_idx
        idx = tm.box_expected(idx, box_with_array)
        result = op(idx, 1)
        tm.assert_equal(result, idx)

    def test_mul_int_array(self, numeric_idx):
        idx = numeric_idx
        didx = idx * idx
        result = idx * np.array(5, dtype='int64')
        tm.assert_index_equal(result, idx * 5)
        arr_dtype = 'uint64' if idx.dtype == np.uint64 else 'int64'
        result = idx * np.arange(5, dtype=arr_dtype)
        tm.assert_index_equal(result, didx)

    def test_mul_int_series(self, numeric_idx):
        idx = numeric_idx
        didx = idx * idx
        arr_dtype = 'uint64' if idx.dtype == np.uint64 else 'int64'
        result = idx * Series(np.arange(5, dtype=arr_dtype))
        tm.assert_series_equal(result, Series(didx))

    def test_mul_float_series(self, numeric_idx):
        idx = numeric_idx
        rng5 = np.arange(5, dtype='float64')
        result = idx * Series(rng5 + 0.1)
        expected = Series(rng5 * (rng5 + 0.1))
        tm.assert_series_equal(result, expected)

    def test_mul_index(self, numeric_idx):
        idx = numeric_idx
        result = idx * idx
        tm.assert_index_equal(result, idx ** 2)

    def test_mul_datelike_raises(self, numeric_idx):
        idx = numeric_idx
        msg = 'cannot perform __rmul__ with this index type'
        with pytest.raises(TypeError, match=msg):
            idx * date_range('20130101', periods=5)

    def test_mul_size_mismatch_raises(self, numeric_idx):
        idx = numeric_idx
        msg = 'operands could not be broadcast together'
        with pytest.raises(ValueError, match=msg):
            idx * idx[0:3]
        with pytest.raises(ValueError, match=msg):
            idx * np.array([1, 2])

    @pytest.mark.parametrize('op', [operator.pow, ops.rpow])
    def test_pow_float(self, op, numeric_idx, box_with_array):
        box = box_with_array
        idx = numeric_idx
        expected = Index(op(idx.values, 2.0))
        idx = tm.box_expected(idx, box)
        expected = tm.box_expected(expected, box)
        result = op(idx, 2.0)
        tm.assert_equal(result, expected)

    def test_modulo(self, numeric_idx, box_with_array):
        box = box_with_array
        idx = numeric_idx
        expected = Index(idx.values % 2)
        idx = tm.box_expected(idx, box)
        expected = tm.box_expected(expected, box)
        result = idx % 2
        tm.assert_equal(result, expected)

    def test_divmod_scalar(self, numeric_idx):
        idx = numeric_idx
        result = divmod(idx, 2)
        with np.errstate(all='ignore'):
            div, mod = divmod(idx.values, 2)
        expected = (Index(div), Index(mod))
        for r, e in zip(result, expected):
            tm.assert_index_equal(r, e)

    def test_divmod_ndarray(self, numeric_idx):
        idx = numeric_idx
        other = np.ones(idx.values.shape, dtype=idx.values.dtype) * 2
        result = divmod(idx, other)
        with np.errstate(all='ignore'):
            div, mod = divmod(idx.values, other)
        expected = (Index(div), Index(mod))
        for r, e in zip(result, expected):
            tm.assert_index_equal(r, e)

    def test_divmod_series(self, numeric_idx):
        idx = numeric_idx
        other = np.ones(idx.values.shape, dtype=idx.values.dtype) * 2
        result = divmod(idx, Series(other))
        with np.errstate(all='ignore'):
            div, mod = divmod(idx.values, other)
        expected = (Series(div), Series(mod))
        for r, e in zip(result, expected):
            tm.assert_series_equal(r, e)

    @pytest.mark.parametrize('other', [np.nan, 7, -23, 2.718, -3.14, np.inf])
    def test_ops_np_scalar(self, other):
        vals = np.random.default_rng(2).standard_normal((5, 3))
        f = lambda x: pd.DataFrame(x, index=list('ABCDE'), columns=['jim', 'joe', 'jolie'])
        df = f(vals)
        tm.assert_frame_equal(df / np.array(other), f(vals / other))
        tm.assert_frame_equal(np.array(other) * df, f(vals * other))
        tm.assert_frame_equal(df + np.array(other), f(vals + other))
        tm.assert_frame_equal(np.array(other) - df, f(other - vals))

    def test_operators_frame(self):
        ts = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10), name='ts')
        ts.name = 'ts'
        df = pd.DataFrame({'A': ts})
        tm.assert_series_equal(ts + ts, ts + df['A'], check_names=False)
        tm.assert_series_equal(ts ** ts, ts ** df['A'], check_names=False)
        tm.assert_series_equal(ts < ts, ts < df['A'], check_names=False)
        tm.assert_series_equal(ts / ts, ts / df['A'], check_names=False)

    def test_modulo2(self):
        with np.errstate(all='ignore'):
            p = pd.DataFrame({'first': [3, 4, 5, 8], 'second': [0, 0, 0, 3]})
            result = p['first'] % p['second']
            expected = Series(p['first'].values % p['second'].values, dtype='float64')
            expected.iloc[0:3] = np.nan
            tm.assert_series_equal(result, expected)
            result = p['first'] % 0
            expected = Series(np.nan, index=p.index, name='first')
            tm.assert_series_equal(result, expected)
            p = p.astype('float64')
            result = p['first'] % p['second']
            expected = Series(p['first'].values % p['second'].values)
            tm.assert_series_equal(result, expected)
            p = p.astype('float64')
            result = p['first'] % p['second']
            result2 = p['second'] % p['first']
            assert not result.equals(result2)

    def test_modulo_zero_int(self):
        with np.errstate(all='ignore'):
            s = Series([0, 1])
            result = s % 0
            expected = Series([np.nan, np.nan])
            tm.assert_series_equal(result, expected)
            result = 0 % s
            expected = Series([np.nan, 0.0])
            tm.assert_series_equal(result, expected)