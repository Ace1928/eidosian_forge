import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
class TestDataFrameEvalWithFrame:

    @pytest.fixture
    def frame(self):
        return DataFrame(np.random.default_rng(2).standard_normal((10, 3)), columns=list('abc'))

    def test_simple_expr(self, frame, parser, engine):
        res = frame.eval('a + b', engine=engine, parser=parser)
        expect = frame.a + frame.b
        tm.assert_series_equal(res, expect)

    def test_bool_arith_expr(self, frame, parser, engine):
        res = frame.eval('a[a < 1] + b', engine=engine, parser=parser)
        expect = frame.a[frame.a < 1] + frame.b
        tm.assert_series_equal(res, expect)

    @pytest.mark.parametrize('op', ['+', '-', '*', '/'])
    def test_invalid_type_for_operator_raises(self, parser, engine, op):
        df = DataFrame({'a': [1, 2], 'b': ['c', 'd']})
        msg = "unsupported operand type\\(s\\) for .+: '.+' and '.+'|Cannot"
        with pytest.raises(TypeError, match=msg):
            df.eval(f'a {op} b', engine=engine, parser=parser)