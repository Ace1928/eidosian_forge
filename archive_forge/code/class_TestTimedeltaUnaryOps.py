from datetime import timedelta
import sys
from hypothesis import (
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsTimedelta
from pandas import (
import pandas._testing as tm
class TestTimedeltaUnaryOps:

    def test_invert(self):
        td = Timedelta(10, unit='d')
        msg = 'bad operand type for unary ~'
        with pytest.raises(TypeError, match=msg):
            ~td
        with pytest.raises(TypeError, match=msg):
            ~td.to_pytimedelta()
        umsg = "ufunc 'invert' not supported for the input types"
        with pytest.raises(TypeError, match=umsg):
            ~td.to_timedelta64()

    def test_unary_ops(self):
        td = Timedelta(10, unit='d')
        assert -td == Timedelta(-10, unit='d')
        assert -td == Timedelta('-10d')
        assert +td == Timedelta(10, unit='d')
        assert abs(td) == td
        assert abs(-td) == td
        assert abs(-td) == Timedelta('10d')