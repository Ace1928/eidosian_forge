from statsmodels.compat.python import lrange
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy.linalg import toeplitz
from scipy.stats import t as student_t
from statsmodels.datasets import longley
from statsmodels.regression.linear_model import (
from statsmodels.tools.tools import add_constant
class TestFtest:
    """
    Tests f_test vs. RegressionResults
    """

    @classmethod
    def setup_class(cls):
        data = longley.load()
        data.exog = add_constant(data.exog, prepend=False)
        cls.res1 = OLS(data.endog, data.exog).fit()
        R = np.identity(7)[:-1, :]
        cls.Ftest = cls.res1.f_test(R)

    def test_F(self):
        assert_almost_equal(self.Ftest.fvalue, self.res1.fvalue, DECIMAL_4)

    def test_p(self):
        assert_almost_equal(self.Ftest.pvalue, self.res1.f_pvalue, DECIMAL_4)

    def test_Df_denom(self):
        assert_equal(self.Ftest.df_denom, self.res1.model.df_resid)

    def test_Df_num(self):
        assert_equal(self.Ftest.df_num, 6)