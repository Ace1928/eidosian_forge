import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
@td.skip_if_no('numexpr')
class TestDataFrameQueryNumExprPython(TestDataFrameQueryNumExprPandas):

    @pytest.fixture
    def engine(self):
        return 'numexpr'

    @pytest.fixture
    def parser(self):
        return 'python'

    def test_date_query_no_attribute_access(self, engine, parser):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        df['dates1'] = date_range('1/1/2012', periods=5)
        df['dates2'] = date_range('1/1/2013', periods=5)
        df['dates3'] = date_range('1/1/2014', periods=5)
        res = df.query('(dates1 < 20130101) & (20130101 < dates3)', engine=engine, parser=parser)
        expec = df[(df.dates1 < '20130101') & ('20130101' < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_query_with_NaT(self, engine, parser):
        n = 10
        df = DataFrame(np.random.default_rng(2).standard_normal((n, 3)))
        df['dates1'] = date_range('1/1/2012', periods=n)
        df['dates2'] = date_range('1/1/2013', periods=n)
        df['dates3'] = date_range('1/1/2014', periods=n)
        df.loc[np.random.default_rng(2).random(n) > 0.5, 'dates1'] = pd.NaT
        df.loc[np.random.default_rng(2).random(n) > 0.5, 'dates3'] = pd.NaT
        res = df.query('(dates1 < 20130101) & (20130101 < dates3)', engine=engine, parser=parser)
        expec = df[(df.dates1 < '20130101') & ('20130101' < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_index_query(self, engine, parser):
        n = 10
        df = DataFrame(np.random.default_rng(2).standard_normal((n, 3)))
        df['dates1'] = date_range('1/1/2012', periods=n)
        df['dates3'] = date_range('1/1/2014', periods=n)
        return_value = df.set_index('dates1', inplace=True, drop=True)
        assert return_value is None
        res = df.query('(index < 20130101) & (20130101 < dates3)', engine=engine, parser=parser)
        expec = df[(df.index < '20130101') & ('20130101' < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_index_query_with_NaT(self, engine, parser):
        n = 10
        df = DataFrame(np.random.default_rng(2).standard_normal((n, 3))).astype({0: object})
        df['dates1'] = date_range('1/1/2012', periods=n)
        df['dates3'] = date_range('1/1/2014', periods=n)
        df.iloc[0, 0] = pd.NaT
        return_value = df.set_index('dates1', inplace=True, drop=True)
        assert return_value is None
        res = df.query('(index < 20130101) & (20130101 < dates3)', engine=engine, parser=parser)
        expec = df[(df.index < '20130101') & ('20130101' < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_index_query_with_NaT_duplicates(self, engine, parser):
        n = 10
        df = DataFrame(np.random.default_rng(2).standard_normal((n, 3)))
        df['dates1'] = date_range('1/1/2012', periods=n)
        df['dates3'] = date_range('1/1/2014', periods=n)
        df.loc[np.random.default_rng(2).random(n) > 0.5, 'dates1'] = pd.NaT
        return_value = df.set_index('dates1', inplace=True, drop=True)
        assert return_value is None
        msg = "'BoolOp' nodes are not implemented"
        with pytest.raises(NotImplementedError, match=msg):
            df.query('index < 20130101 < dates3', engine=engine, parser=parser)

    def test_nested_scope(self, engine, parser):
        x = 1
        result = pd.eval('x + 1', engine=engine, parser=parser)
        assert result == 2
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        df2 = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        msg = "The '@' prefix is only supported by the pandas parser"
        with pytest.raises(SyntaxError, match=msg):
            df.query('(@df>0) & (@df2>0)', engine=engine, parser=parser)
        with pytest.raises(UndefinedVariableError, match="name 'df' is not defined"):
            df.query('(df>0) & (df2>0)', engine=engine, parser=parser)
        expected = df[(df > 0) & (df2 > 0)]
        result = pd.eval('df[(df > 0) & (df2 > 0)]', engine=engine, parser=parser)
        tm.assert_frame_equal(expected, result)
        expected = df[(df > 0) & (df2 > 0) & (df[df > 0] > 0)]
        result = pd.eval('df[(df > 0) & (df2 > 0) & (df[df > 0] > 0)]', engine=engine, parser=parser)
        tm.assert_frame_equal(expected, result)

    def test_query_numexpr_with_min_and_max_columns(self):
        df = DataFrame({'min': [1, 2, 3], 'max': [4, 5, 6]})
        regex_to_match = 'Variables in expression \\"\\(min\\) == \\(1\\)\\" overlap with builtins: \\(\'min\'\\)'
        with pytest.raises(NumExprClobberingError, match=regex_to_match):
            df.query('min == 1')
        regex_to_match = 'Variables in expression \\"\\(max\\) == \\(1\\)\\" overlap with builtins: \\(\'max\'\\)'
        with pytest.raises(NumExprClobberingError, match=regex_to_match):
            df.query('max == 1')