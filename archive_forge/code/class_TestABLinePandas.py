import numpy as np
from numpy.testing import assert_array_less, assert_equal, assert_raises
from pandas import DataFrame, Series
import pytest
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import (
class TestABLinePandas(TestABLine):

    @classmethod
    def setup_class(cls):
        np.random.seed(12345)
        X = sm.add_constant(np.random.normal(0, 20, size=30))
        y = np.dot(X, [25, 3.5]) + np.random.normal(0, 30, size=30)
        cls.X = X
        cls.y = y
        X = DataFrame(X, columns=['const', 'someX'])
        y = Series(y, name='outcome')
        mod = sm.OLS(y, X).fit()
        cls.mod = mod