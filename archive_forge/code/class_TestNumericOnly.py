import builtins
from io import StringIO
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.tests.groupby import get_groupby_method_args
from pandas.util import _test_decorators as td
class TestNumericOnly:

    @pytest.fixture
    def df(self):
        df = DataFrame({'group': [1, 1, 2], 'int': [1, 2, 3], 'float': [4.0, 5.0, 6.0], 'string': list('abc'), 'category_string': Series(list('abc')).astype('category'), 'category_int': [7, 8, 9], 'datetime': date_range('20130101', periods=3), 'datetimetz': date_range('20130101', periods=3, tz='US/Eastern'), 'timedelta': pd.timedelta_range('1 s', periods=3, freq='s')}, columns=['group', 'int', 'float', 'string', 'category_string', 'category_int', 'datetime', 'datetimetz', 'timedelta'])
        return df

    @pytest.mark.parametrize('method', ['mean', 'median'])
    def test_averages(self, df, method):
        expected_columns_numeric = Index(['int', 'float', 'category_int'])
        gb = df.groupby('group')
        expected = DataFrame({'category_int': [7.5, 9], 'float': [4.5, 6.0], 'timedelta': [pd.Timedelta('1.5s'), pd.Timedelta('3s')], 'int': [1.5, 3], 'datetime': [Timestamp('2013-01-01 12:00:00'), Timestamp('2013-01-03 00:00:00')], 'datetimetz': [Timestamp('2013-01-01 12:00:00', tz='US/Eastern'), Timestamp('2013-01-03 00:00:00', tz='US/Eastern')]}, index=Index([1, 2], name='group'), columns=['int', 'float', 'category_int'])
        result = getattr(gb, method)(numeric_only=True)
        tm.assert_frame_equal(result.reindex_like(expected), expected)
        expected_columns = expected.columns
        self._check(df, method, expected_columns, expected_columns_numeric)

    @pytest.mark.parametrize('method', ['min', 'max'])
    def test_extrema(self, df, method):
        expected_columns = Index(['int', 'float', 'string', 'category_int', 'datetime', 'datetimetz', 'timedelta'])
        expected_columns_numeric = expected_columns
        self._check(df, method, expected_columns, expected_columns_numeric)

    @pytest.mark.parametrize('method', ['first', 'last'])
    def test_first_last(self, df, method):
        expected_columns = Index(['int', 'float', 'string', 'category_string', 'category_int', 'datetime', 'datetimetz', 'timedelta'])
        expected_columns_numeric = expected_columns
        self._check(df, method, expected_columns, expected_columns_numeric)

    @pytest.mark.parametrize('method', ['sum', 'cumsum'])
    def test_sum_cumsum(self, df, method):
        expected_columns_numeric = Index(['int', 'float', 'category_int'])
        expected_columns = Index(['int', 'float', 'string', 'category_int', 'timedelta'])
        if method == 'cumsum':
            expected_columns = Index(['int', 'float', 'category_int', 'timedelta'])
        self._check(df, method, expected_columns, expected_columns_numeric)

    @pytest.mark.parametrize('method', ['prod', 'cumprod'])
    def test_prod_cumprod(self, df, method):
        expected_columns = Index(['int', 'float', 'category_int'])
        expected_columns_numeric = expected_columns
        self._check(df, method, expected_columns, expected_columns_numeric)

    @pytest.mark.parametrize('method', ['cummin', 'cummax'])
    def test_cummin_cummax(self, df, method):
        expected_columns = Index(['int', 'float', 'category_int', 'datetime', 'datetimetz', 'timedelta'])
        expected_columns_numeric = expected_columns
        self._check(df, method, expected_columns, expected_columns_numeric)

    def _check(self, df, method, expected_columns, expected_columns_numeric):
        gb = df.groupby('group')
        exception = NotImplementedError if method.startswith('cum') else TypeError
        if method in ('min', 'max', 'cummin', 'cummax', 'cumsum', 'cumprod'):
            msg = '|'.join(['Categorical is not ordered', 'function is not implemented for this dtype', f'Cannot perform {method} with non-ordered Categorical'])
            with pytest.raises(exception, match=msg):
                getattr(gb, method)()
        elif method in ('sum', 'mean', 'median', 'prod'):
            msg = '|'.join(['category type does not support sum operations', '[Cc]ould not convert', "can't multiply sequence by non-int of type 'str'"])
            with pytest.raises(exception, match=msg):
                getattr(gb, method)()
        else:
            result = getattr(gb, method)()
            tm.assert_index_equal(result.columns, expected_columns_numeric)
        if method not in ('first', 'last'):
            msg = '|'.join(['[Cc]ould not convert', 'Categorical is not ordered', 'category type does not support', "can't multiply sequence", 'function is not implemented for this dtype', f'Cannot perform {method} with non-ordered Categorical'])
            with pytest.raises(exception, match=msg):
                getattr(gb, method)(numeric_only=False)
        else:
            result = getattr(gb, method)(numeric_only=False)
            tm.assert_index_equal(result.columns, expected_columns)