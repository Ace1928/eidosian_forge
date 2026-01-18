from collections import deque
from datetime import (
from enum import Enum
import functools
import operator
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation import expressions as expr
from pandas.tests.frame.common import (
class TestFrameFlexComparisons:

    @pytest.mark.parametrize('op', ['eq', 'ne', 'gt', 'lt', 'ge', 'le'])
    def test_bool_flex_frame(self, op):
        data = np.random.default_rng(2).standard_normal((5, 3))
        other_data = np.random.default_rng(2).standard_normal((5, 3))
        df = DataFrame(data)
        other = DataFrame(other_data)
        ndim_5 = np.ones(df.shape + (1, 3))
        assert df.eq(df).values.all()
        assert not df.ne(df).values.any()
        f = getattr(df, op)
        o = getattr(operator, op)
        tm.assert_frame_equal(f(other), o(df, other))
        part_o = other.loc[3:, 1:].copy()
        rs = f(part_o)
        xp = o(df, part_o.reindex(index=df.index, columns=df.columns))
        tm.assert_frame_equal(rs, xp)
        tm.assert_frame_equal(f(other.values), o(df, other.values))
        tm.assert_frame_equal(f(0), o(df, 0))
        msg = 'Unable to coerce to Series/DataFrame'
        tm.assert_frame_equal(f(np.nan), o(df, np.nan))
        with pytest.raises(ValueError, match=msg):
            f(ndim_5)

    @pytest.mark.parametrize('box', [np.array, Series])
    def test_bool_flex_series(self, box):
        data = np.random.default_rng(2).standard_normal((5, 3))
        df = DataFrame(data)
        idx_ser = box(np.random.default_rng(2).standard_normal(5))
        col_ser = box(np.random.default_rng(2).standard_normal(3))
        idx_eq = df.eq(idx_ser, axis=0)
        col_eq = df.eq(col_ser)
        idx_ne = df.ne(idx_ser, axis=0)
        col_ne = df.ne(col_ser)
        tm.assert_frame_equal(col_eq, df == Series(col_ser))
        tm.assert_frame_equal(col_eq, -col_ne)
        tm.assert_frame_equal(idx_eq, -idx_ne)
        tm.assert_frame_equal(idx_eq, df.T.eq(idx_ser).T)
        tm.assert_frame_equal(col_eq, df.eq(list(col_ser)))
        tm.assert_frame_equal(idx_eq, df.eq(Series(idx_ser), axis=0))
        tm.assert_frame_equal(idx_eq, df.eq(list(idx_ser), axis=0))
        idx_gt = df.gt(idx_ser, axis=0)
        col_gt = df.gt(col_ser)
        idx_le = df.le(idx_ser, axis=0)
        col_le = df.le(col_ser)
        tm.assert_frame_equal(col_gt, df > Series(col_ser))
        tm.assert_frame_equal(col_gt, -col_le)
        tm.assert_frame_equal(idx_gt, -idx_le)
        tm.assert_frame_equal(idx_gt, df.T.gt(idx_ser).T)
        idx_ge = df.ge(idx_ser, axis=0)
        col_ge = df.ge(col_ser)
        idx_lt = df.lt(idx_ser, axis=0)
        col_lt = df.lt(col_ser)
        tm.assert_frame_equal(col_ge, df >= Series(col_ser))
        tm.assert_frame_equal(col_ge, -col_lt)
        tm.assert_frame_equal(idx_ge, -idx_lt)
        tm.assert_frame_equal(idx_ge, df.T.ge(idx_ser).T)
        idx_ser = Series(np.random.default_rng(2).standard_normal(5))
        col_ser = Series(np.random.default_rng(2).standard_normal(3))

    def test_bool_flex_frame_na(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        df.loc[0, 0] = np.nan
        rs = df.eq(df)
        assert not rs.loc[0, 0]
        rs = df.ne(df)
        assert rs.loc[0, 0]
        rs = df.gt(df)
        assert not rs.loc[0, 0]
        rs = df.lt(df)
        assert not rs.loc[0, 0]
        rs = df.ge(df)
        assert not rs.loc[0, 0]
        rs = df.le(df)
        assert not rs.loc[0, 0]

    def test_bool_flex_frame_complex_dtype(self):
        arr = np.array([np.nan, 1, 6, np.nan])
        arr2 = np.array([2j, np.nan, 7, None])
        df = DataFrame({'a': arr})
        df2 = DataFrame({'a': arr2})
        msg = '|'.join(["'>' not supported between instances of '.*' and 'complex'", 'unorderable types: .*complex\\(\\)'])
        with pytest.raises(TypeError, match=msg):
            df.gt(df2)
        with pytest.raises(TypeError, match=msg):
            df['a'].gt(df2['a'])
        with pytest.raises(TypeError, match=msg):
            df.values > df2.values
        rs = df.ne(df2)
        assert rs.values.all()
        arr3 = np.array([2j, np.nan, None])
        df3 = DataFrame({'a': arr3})
        with pytest.raises(TypeError, match=msg):
            df3.gt(2j)
        with pytest.raises(TypeError, match=msg):
            df3['a'].gt(2j)
        with pytest.raises(TypeError, match=msg):
            df3.values > 2j

    def test_bool_flex_frame_object_dtype(self):
        df1 = DataFrame({'col': ['foo', np.nan, 'bar']}, dtype=object)
        df2 = DataFrame({'col': ['foo', datetime.now(), 'bar']}, dtype=object)
        result = df1.ne(df2)
        exp = DataFrame({'col': [False, True, False]})
        tm.assert_frame_equal(result, exp)

    def test_flex_comparison_nat(self):
        df = DataFrame([pd.NaT])
        result = df == pd.NaT
        assert result.iloc[0, 0].item() is False
        result = df.eq(pd.NaT)
        assert result.iloc[0, 0].item() is False
        result = df != pd.NaT
        assert result.iloc[0, 0].item() is True
        result = df.ne(pd.NaT)
        assert result.iloc[0, 0].item() is True

    @pytest.mark.parametrize('opname', ['eq', 'ne', 'gt', 'lt', 'ge', 'le'])
    def test_df_flex_cmp_constant_return_types(self, opname):
        df = DataFrame({'x': [1, 2, 3], 'y': [1.0, 2.0, 3.0]})
        const = 2
        result = getattr(df, opname)(const).dtypes.value_counts()
        tm.assert_series_equal(result, Series([2], index=[np.dtype(bool)], name='count'))

    @pytest.mark.parametrize('opname', ['eq', 'ne', 'gt', 'lt', 'ge', 'le'])
    def test_df_flex_cmp_constant_return_types_empty(self, opname):
        df = DataFrame({'x': [1, 2, 3], 'y': [1.0, 2.0, 3.0]})
        const = 2
        empty = df.iloc[:0]
        result = getattr(empty, opname)(const).dtypes.value_counts()
        tm.assert_series_equal(result, Series([2], index=[np.dtype(bool)], name='count'))

    def test_df_flex_cmp_ea_dtype_with_ndarray_series(self):
        ii = pd.IntervalIndex.from_breaks([1, 2, 3])
        df = DataFrame({'A': ii, 'B': ii})
        ser = Series([0, 0])
        res = df.eq(ser, axis=0)
        expected = DataFrame({'A': [False, False], 'B': [False, False]})
        tm.assert_frame_equal(res, expected)
        ser2 = Series([1, 2], index=['A', 'B'])
        res2 = df.eq(ser2, axis=1)
        tm.assert_frame_equal(res2, expected)