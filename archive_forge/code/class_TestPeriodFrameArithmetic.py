import operator
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import TimedeltaArray
from pandas.tests.arithmetic.common import (
class TestPeriodFrameArithmetic:

    def test_ops_frame_period(self):
        df = pd.DataFrame({'A': [Period('2015-01', freq='M'), Period('2015-02', freq='M')], 'B': [Period('2014-01', freq='M'), Period('2014-02', freq='M')]})
        assert df['A'].dtype == 'Period[M]'
        assert df['B'].dtype == 'Period[M]'
        p = Period('2015-03', freq='M')
        off = p.freq
        exp = pd.DataFrame({'A': np.array([2 * off, 1 * off], dtype=object), 'B': np.array([14 * off, 13 * off], dtype=object)})
        tm.assert_frame_equal(p - df, exp)
        tm.assert_frame_equal(df - p, -1 * exp)
        df2 = pd.DataFrame({'A': [Period('2015-05', freq='M'), Period('2015-06', freq='M')], 'B': [Period('2015-05', freq='M'), Period('2015-06', freq='M')]})
        assert df2['A'].dtype == 'Period[M]'
        assert df2['B'].dtype == 'Period[M]'
        exp = pd.DataFrame({'A': np.array([4 * off, 4 * off], dtype=object), 'B': np.array([16 * off, 16 * off], dtype=object)})
        tm.assert_frame_equal(df2 - df, exp)
        tm.assert_frame_equal(df - df2, -1 * exp)