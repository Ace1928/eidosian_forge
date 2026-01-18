import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
class TestDataFrameQueryBacktickQuoting:

    @pytest.fixture
    def df(self):
        """
        Yields a dataframe with strings that may or may not need escaping
        by backticks. The last two columns cannot be escaped by backticks
        and should raise a ValueError.
        """
        yield DataFrame({'A': [1, 2, 3], 'B B': [3, 2, 1], 'C C': [4, 5, 6], 'C  C': [7, 4, 3], 'C_C': [8, 9, 10], 'D_D D': [11, 1, 101], 'E.E': [6, 3, 5], 'F-F': [8, 1, 10], '1e1': [2, 4, 8], 'def': [10, 11, 2], 'A (x)': [4, 1, 3], 'B(x)': [1, 1, 5], 'B (x)': [2, 7, 4], "  &^ :!€$?(} >    <++*''  ": [2, 5, 6], '': [10, 11, 1], ' A': [4, 7, 9], '  ': [1, 2, 1], "it's": [6, 3, 1], "that's": [9, 1, 8], '☺': [8, 7, 6], 'foo#bar': [2, 4, 5], 1: [5, 7, 9]})

    def test_single_backtick_variable_query(self, df):
        res = df.query('1 < `B B`')
        expect = df[1 < df['B B']]
        tm.assert_frame_equal(res, expect)

    def test_two_backtick_variables_query(self, df):
        res = df.query('1 < `B B` and 4 < `C C`')
        expect = df[(1 < df['B B']) & (4 < df['C C'])]
        tm.assert_frame_equal(res, expect)

    def test_single_backtick_variable_expr(self, df):
        res = df.eval('A + `B B`')
        expect = df['A'] + df['B B']
        tm.assert_series_equal(res, expect)

    def test_two_backtick_variables_expr(self, df):
        res = df.eval('`B B` + `C C`')
        expect = df['B B'] + df['C C']
        tm.assert_series_equal(res, expect)

    def test_already_underscore_variable(self, df):
        res = df.eval('`C_C` + A')
        expect = df['C_C'] + df['A']
        tm.assert_series_equal(res, expect)

    def test_same_name_but_underscores(self, df):
        res = df.eval('C_C + `C C`')
        expect = df['C_C'] + df['C C']
        tm.assert_series_equal(res, expect)

    def test_mixed_underscores_and_spaces(self, df):
        res = df.eval('A + `D_D D`')
        expect = df['A'] + df['D_D D']
        tm.assert_series_equal(res, expect)

    def test_backtick_quote_name_with_no_spaces(self, df):
        res = df.eval('A + `C_C`')
        expect = df['A'] + df['C_C']
        tm.assert_series_equal(res, expect)

    def test_special_characters(self, df):
        res = df.eval('`E.E` + `F-F` - A')
        expect = df['E.E'] + df['F-F'] - df['A']
        tm.assert_series_equal(res, expect)

    def test_start_with_digit(self, df):
        res = df.eval('A + `1e1`')
        expect = df['A'] + df['1e1']
        tm.assert_series_equal(res, expect)

    def test_keyword(self, df):
        res = df.eval('A + `def`')
        expect = df['A'] + df['def']
        tm.assert_series_equal(res, expect)

    def test_unneeded_quoting(self, df):
        res = df.query('`A` > 2')
        expect = df[df['A'] > 2]
        tm.assert_frame_equal(res, expect)

    def test_parenthesis(self, df):
        res = df.query('`A (x)` > 2')
        expect = df[df['A (x)'] > 2]
        tm.assert_frame_equal(res, expect)

    def test_empty_string(self, df):
        res = df.query('`` > 5')
        expect = df[df[''] > 5]
        tm.assert_frame_equal(res, expect)

    def test_multiple_spaces(self, df):
        res = df.query('`C  C` > 5')
        expect = df[df['C  C'] > 5]
        tm.assert_frame_equal(res, expect)

    def test_start_with_spaces(self, df):
        res = df.eval('` A` + `  `')
        expect = df[' A'] + df['  ']
        tm.assert_series_equal(res, expect)

    def test_lots_of_operators_string(self, df):
        res = df.query("`  &^ :!€$?(} >    <++*''  ` > 4")
        expect = df[df["  &^ :!€$?(} >    <++*''  "] > 4]
        tm.assert_frame_equal(res, expect)

    def test_missing_attribute(self, df):
        message = "module 'pandas' has no attribute 'thing'"
        with pytest.raises(AttributeError, match=message):
            df.eval('@pd.thing')

    def test_failing_quote(self, df):
        msg = '(Could not convert ).*( to a valid Python identifier.)'
        with pytest.raises(SyntaxError, match=msg):
            df.query("`it's` > `that's`")

    def test_failing_character_outside_range(self, df):
        msg = '(Could not convert ).*( to a valid Python identifier.)'
        with pytest.raises(SyntaxError, match=msg):
            df.query('`☺` > 4')

    def test_failing_hashtag(self, df):
        msg = 'Failed to parse backticks'
        with pytest.raises(SyntaxError, match=msg):
            df.query('`foo#bar` > 4')

    def test_call_non_named_expression(self, df):
        """
        Only attributes and variables ('named functions') can be called.
        .__call__() is not an allowed attribute because that would allow
        calling anything.
        https://github.com/pandas-dev/pandas/pull/32460
        """

        def func(*_):
            return 1
        funcs = [func]
        df.eval('@func()')
        with pytest.raises(TypeError, match='Only named functions are supported'):
            df.eval('@funcs[0]()')
        with pytest.raises(TypeError, match='Only named functions are supported'):
            df.eval('@funcs[0].__call__()')

    def test_ea_dtypes(self, any_numeric_ea_and_arrow_dtype):
        df = DataFrame([[1, 2], [3, 4]], columns=['a', 'b'], dtype=any_numeric_ea_and_arrow_dtype)
        warning = RuntimeWarning if NUMEXPR_INSTALLED else None
        with tm.assert_produces_warning(warning):
            result = df.eval('c = b - a')
        expected = DataFrame([[1, 2, 1], [3, 4, 1]], columns=['a', 'b', 'c'], dtype=any_numeric_ea_and_arrow_dtype)
        tm.assert_frame_equal(result, expected)

    def test_ea_dtypes_and_scalar(self):
        df = DataFrame([[1, 2], [3, 4]], columns=['a', 'b'], dtype='Float64')
        warning = RuntimeWarning if NUMEXPR_INSTALLED else None
        with tm.assert_produces_warning(warning):
            result = df.eval('c = b - 1')
        expected = DataFrame([[1, 2, 1], [3, 4, 3]], columns=['a', 'b', 'c'], dtype='Float64')
        tm.assert_frame_equal(result, expected)

    def test_ea_dtypes_and_scalar_operation(self, any_numeric_ea_and_arrow_dtype):
        df = DataFrame([[1, 2], [3, 4]], columns=['a', 'b'], dtype=any_numeric_ea_and_arrow_dtype)
        result = df.eval('c = 2 - 1')
        expected = DataFrame({'a': Series([1, 3], dtype=any_numeric_ea_and_arrow_dtype), 'b': Series([2, 4], dtype=any_numeric_ea_and_arrow_dtype), 'c': Series([1, 1], dtype=result['c'].dtype)})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('dtype', ['int64', 'Int64', 'int64[pyarrow]'])
    def test_query_ea_dtypes(self, dtype):
        if dtype == 'int64[pyarrow]':
            pytest.importorskip('pyarrow')
        df = DataFrame({'a': Series([1, 2], dtype=dtype)})
        ref = {2}
        warning = RuntimeWarning if dtype == 'Int64' and NUMEXPR_INSTALLED else None
        with tm.assert_produces_warning(warning):
            result = df.query('a in @ref')
        expected = DataFrame({'a': Series([2], dtype=dtype, index=[1])})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('engine', ['python', 'numexpr'])
    @pytest.mark.parametrize('dtype', ['int64', 'Int64', 'int64[pyarrow]'])
    def test_query_ea_equality_comparison(self, dtype, engine):
        warning = RuntimeWarning if engine == 'numexpr' else None
        if engine == 'numexpr' and (not NUMEXPR_INSTALLED):
            pytest.skip('numexpr not installed')
        if dtype == 'int64[pyarrow]':
            pytest.importorskip('pyarrow')
        df = DataFrame({'A': Series([1, 1, 2], dtype='Int64'), 'B': Series([1, 2, 2], dtype=dtype)})
        with tm.assert_produces_warning(warning):
            result = df.query('A == B', engine=engine)
        expected = DataFrame({'A': Series([1, 2], dtype='Int64', index=[0, 2]), 'B': Series([1, 2], dtype=dtype, index=[0, 2])})
        tm.assert_frame_equal(result, expected)

    def test_all_nat_in_object(self):
        now = pd.Timestamp.now('UTC')
        df = DataFrame({'a': pd.to_datetime([None, None], utc=True)}, dtype=object)
        result = df.query('a > @now')
        expected = DataFrame({'a': []}, dtype=object)
        tm.assert_frame_equal(result, expected)