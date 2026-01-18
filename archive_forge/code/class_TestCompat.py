import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
class TestCompat:

    @pytest.fixture
    def df(self):
        return DataFrame({'A': [1, 2, 3]})

    @pytest.fixture
    def expected1(self, df):
        return df[df.A > 0]

    @pytest.fixture
    def expected2(self, df):
        return df.A + 1

    def test_query_default(self, df, expected1, expected2):
        result = df.query('A>0')
        tm.assert_frame_equal(result, expected1)
        result = df.eval('A+1')
        tm.assert_series_equal(result, expected2, check_names=False)

    def test_query_None(self, df, expected1, expected2):
        result = df.query('A>0', engine=None)
        tm.assert_frame_equal(result, expected1)
        result = df.eval('A+1', engine=None)
        tm.assert_series_equal(result, expected2, check_names=False)

    def test_query_python(self, df, expected1, expected2):
        result = df.query('A>0', engine='python')
        tm.assert_frame_equal(result, expected1)
        result = df.eval('A+1', engine='python')
        tm.assert_series_equal(result, expected2, check_names=False)

    def test_query_numexpr(self, df, expected1, expected2):
        if NUMEXPR_INSTALLED:
            result = df.query('A>0', engine='numexpr')
            tm.assert_frame_equal(result, expected1)
            result = df.eval('A+1', engine='numexpr')
            tm.assert_series_equal(result, expected2, check_names=False)
        else:
            msg = "'numexpr' is not installed or an unsupported version. Cannot use engine='numexpr' for query/eval if 'numexpr' is not installed"
            with pytest.raises(ImportError, match=msg):
                df.query('A>0', engine='numexpr')
            with pytest.raises(ImportError, match=msg):
                df.eval('A+1', engine='numexpr')