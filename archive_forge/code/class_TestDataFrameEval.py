import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
class TestDataFrameEval:

    @pytest.mark.parametrize('n', [4, 4000])
    @pytest.mark.parametrize('op_str,op,rop', [('+', '__add__', '__radd__'), ('-', '__sub__', '__rsub__'), ('*', '__mul__', '__rmul__'), ('/', '__truediv__', '__rtruediv__')])
    def test_ops(self, op_str, op, rop, n):
        df = DataFrame(1, index=range(n), columns=list('abcd'))
        df.iloc[0] = 2
        m = df.mean()
        base = DataFrame(np.tile(m.values, n).reshape(n, -1), columns=list('abcd'))
        expected = eval(f'base {op_str} df')
        result = eval(f'm {op_str} df')
        tm.assert_frame_equal(result, expected)
        if op in ['+', '*']:
            result = getattr(df, op)(m)
            tm.assert_frame_equal(result, expected)
        elif op in ['-', '/']:
            result = getattr(df, rop)(m)
            tm.assert_frame_equal(result, expected)

    def test_dataframe_sub_numexpr_path(self):
        df = DataFrame({'A': np.random.default_rng(2).standard_normal(25000)})
        df.iloc[0:5] = np.nan
        expected = 1 - np.isnan(df.iloc[0:25])
        result = (1 - np.isnan(df)).iloc[0:25]
        tm.assert_frame_equal(result, expected)

    def test_query_non_str(self):
        df = DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'b']})
        msg = 'expr must be a string to be evaluated'
        with pytest.raises(ValueError, match=msg):
            df.query(lambda x: x.B == 'b')
        with pytest.raises(ValueError, match=msg):
            df.query(111)

    def test_query_empty_string(self):
        df = DataFrame({'A': [1, 2, 3]})
        msg = 'expr cannot be an empty string'
        with pytest.raises(ValueError, match=msg):
            df.query('')

    def test_eval_resolvers_as_list(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), columns=list('ab'))
        dict1 = {'a': 1}
        dict2 = {'b': 2}
        assert df.eval('a + b', resolvers=[dict1, dict2]) == dict1['a'] + dict2['b']
        assert pd.eval('a + b', resolvers=[dict1, dict2]) == dict1['a'] + dict2['b']

    def test_eval_resolvers_combined(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), columns=list('ab'))
        dict1 = {'c': 2}
        result = df.eval('a + b * c', resolvers=[dict1])
        expected = df['a'] + df['b'] * dict1['c']
        tm.assert_series_equal(result, expected)

    def test_eval_object_dtype_binop(self):
        df = DataFrame({'a1': ['Y', 'N']})
        res = df.eval("c = ((a1 == 'Y') & True)")
        expected = DataFrame({'a1': ['Y', 'N'], 'c': [True, False]})
        tm.assert_frame_equal(res, expected)