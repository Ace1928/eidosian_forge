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
class TestFrameComparisons:

    def test_comparison_with_categorical_dtype(self):
        df = DataFrame({'A': ['foo', 'bar', 'baz']})
        exp = DataFrame({'A': [True, False, False]})
        res = df == 'foo'
        tm.assert_frame_equal(res, exp)
        df['A'] = df['A'].astype('category')
        res = df == 'foo'
        tm.assert_frame_equal(res, exp)

    def test_frame_in_list(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((6, 4)), columns=list('ABCD'))
        msg = 'The truth value of a DataFrame is ambiguous'
        with pytest.raises(ValueError, match=msg):
            df in [None]

    @pytest.mark.parametrize('arg, arg2', [[{'a': np.random.default_rng(2).integers(10, size=10), 'b': pd.date_range('20010101', periods=10)}, {'a': np.random.default_rng(2).integers(10, size=10), 'b': np.random.default_rng(2).integers(10, size=10)}], [{'a': np.random.default_rng(2).integers(10, size=10), 'b': np.random.default_rng(2).integers(10, size=10)}, {'a': np.random.default_rng(2).integers(10, size=10), 'b': pd.date_range('20010101', periods=10)}], [{'a': pd.date_range('20010101', periods=10), 'b': pd.date_range('20010101', periods=10)}, {'a': np.random.default_rng(2).integers(10, size=10), 'b': np.random.default_rng(2).integers(10, size=10)}], [{'a': np.random.default_rng(2).integers(10, size=10), 'b': pd.date_range('20010101', periods=10)}, {'a': pd.date_range('20010101', periods=10), 'b': pd.date_range('20010101', periods=10)}]])
    def test_comparison_invalid(self, arg, arg2):
        x = DataFrame(arg)
        y = DataFrame(arg2)
        result = x == y
        expected = DataFrame({col: x[col] == y[col] for col in x.columns}, index=x.index, columns=x.columns)
        tm.assert_frame_equal(result, expected)
        result = x != y
        expected = DataFrame({col: x[col] != y[col] for col in x.columns}, index=x.index, columns=x.columns)
        tm.assert_frame_equal(result, expected)
        msgs = ['Invalid comparison between dtype=datetime64\\[ns\\] and ndarray', 'invalid type promotion', "The DTypes <class 'numpy.dtype\\[.*\\]'> and <class 'numpy.dtype\\[.*\\]'> do not have a common DType."]
        msg = '|'.join(msgs)
        with pytest.raises(TypeError, match=msg):
            x >= y
        with pytest.raises(TypeError, match=msg):
            x > y
        with pytest.raises(TypeError, match=msg):
            x < y
        with pytest.raises(TypeError, match=msg):
            x <= y

    @pytest.mark.parametrize('left, right', [('gt', 'lt'), ('lt', 'gt'), ('ge', 'le'), ('le', 'ge'), ('eq', 'eq'), ('ne', 'ne')])
    def test_timestamp_compare(self, left, right):
        df = DataFrame({'dates1': pd.date_range('20010101', periods=10), 'dates2': pd.date_range('20010102', periods=10), 'intcol': np.random.default_rng(2).integers(1000000000, size=10), 'floatcol': np.random.default_rng(2).standard_normal(10), 'stringcol': [chr(100 + i) for i in range(10)]})
        df.loc[np.random.default_rng(2).random(len(df)) > 0.5, 'dates2'] = pd.NaT
        left_f = getattr(operator, left)
        right_f = getattr(operator, right)
        if left in ['eq', 'ne']:
            expected = left_f(df, pd.Timestamp('20010109'))
            result = right_f(pd.Timestamp('20010109'), df)
            tm.assert_frame_equal(result, expected)
        else:
            msg = "'(<|>)=?' not supported between instances of 'numpy.ndarray' and 'Timestamp'"
            with pytest.raises(TypeError, match=msg):
                left_f(df, pd.Timestamp('20010109'))
            with pytest.raises(TypeError, match=msg):
                right_f(pd.Timestamp('20010109'), df)
        if left in ['eq', 'ne']:
            expected = left_f(df, pd.Timestamp('nat'))
            result = right_f(pd.Timestamp('nat'), df)
            tm.assert_frame_equal(result, expected)
        else:
            msg = "'(<|>)=?' not supported between instances of 'numpy.ndarray' and 'NaTType'"
            with pytest.raises(TypeError, match=msg):
                left_f(df, pd.Timestamp('nat'))
            with pytest.raises(TypeError, match=msg):
                right_f(pd.Timestamp('nat'), df)

    @pytest.mark.xfail(using_pyarrow_string_dtype(), reason="can't compare string and int")
    def test_mixed_comparison(self):
        df = DataFrame([['1989-08-01', 1], ['1989-08-01', 2]])
        other = DataFrame([['a', 'b'], ['c', 'd']])
        result = df == other
        assert not result.any().any()
        result = df != other
        assert result.all().all()

    def test_df_boolean_comparison_error(self):
        df = DataFrame(np.arange(6).reshape((3, 2)))
        expected = DataFrame([[False, False], [True, False], [False, False]])
        result = df == (2, 2)
        tm.assert_frame_equal(result, expected)
        result = df == [2, 2]
        tm.assert_frame_equal(result, expected)

    def test_df_float_none_comparison(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((8, 3)), index=range(8), columns=['A', 'B', 'C'])
        result = df.__eq__(None)
        assert not result.any().any()

    def test_df_string_comparison(self):
        df = DataFrame([{'a': 1, 'b': 'foo'}, {'a': 2, 'b': 'bar'}])
        mask_a = df.a > 1
        tm.assert_frame_equal(df[mask_a], df.loc[1:1, :])
        tm.assert_frame_equal(df[-mask_a], df.loc[0:0, :])
        mask_b = df.b == 'foo'
        tm.assert_frame_equal(df[mask_b], df.loc[0:0, :])
        tm.assert_frame_equal(df[-mask_b], df.loc[1:1, :])