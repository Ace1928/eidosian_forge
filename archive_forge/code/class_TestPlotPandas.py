import numpy as np
from numpy.testing import assert_array_less, assert_equal, assert_raises
from pandas import DataFrame, Series
import pytest
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import (
class TestPlotPandas(TestPlot):

    def setup_method(self):
        nsample = 100
        sig = 0.5
        x1 = np.linspace(0, 20, nsample)
        x2 = 5 + 3 * np.random.randn(nsample)
        X = np.c_[x1, x2, np.sin(0.5 * x1), (x2 - 5) ** 2, np.ones(nsample)]
        beta = [0.5, 0.5, 1, -0.04, 5.0]
        y_true = np.dot(X, beta)
        y = y_true + sig * np.random.normal(size=nsample)
        exog0 = sm.add_constant(np.c_[x1, x2], prepend=False)
        exog0 = DataFrame(exog0, columns=['const', 'var1', 'var2'])
        y = Series(y, name='outcome')
        res = sm.OLS(y, exog0).fit()
        self.res = res
        data = DataFrame(exog0, columns=['const', 'var1', 'var2'])
        data['y'] = y
        self.data = data