import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
class TestDataFrameQueryPythonPandas(TestDataFrameQueryNumExprPandas):

    @pytest.fixture
    def engine(self):
        return 'python'

    @pytest.fixture
    def parser(self):
        return 'pandas'

    def test_query_builtin(self, engine, parser):
        n = m = 10
        df = DataFrame(np.random.default_rng(2).integers(m, size=(n, 3)), columns=list('abc'))
        df.index.name = 'sin'
        expected = df[df.index > 5]
        result = df.query('sin > 5', engine=engine, parser=parser)
        tm.assert_frame_equal(expected, result)