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
class TestTtest2:
    """
    Tests the hypothesis that the coefficients on POP and YEAR
    are equal.

    Results from RPy using 'car' package.
    """

    @classmethod
    def setup_class(cls):
        R = np.zeros(7)
        R[4:6] = [1, -1]
        data = longley.load()
        data.exog = add_constant(data.exog, prepend=False)
        res1 = OLS(data.endog, data.exog).fit()
        cls.Ttest1 = res1.t_test(R)

    def test_tvalue(self):
        assert_almost_equal(self.Ttest1.tvalue, -4.016775463639728, DECIMAL_4)

    def test_sd(self):
        assert_almost_equal(self.Ttest1.sd, 455.39079425195314, DECIMAL_4)

    def test_pvalue(self):
        assert_almost_equal(self.Ttest1.pvalue, 2 * 0.0015163772380932246, DECIMAL_4)

    def test_df_denom(self):
        assert_equal(self.Ttest1.df_denom, 9)

    def test_effect(self):
        assert_almost_equal(self.Ttest1.effect, -1829.2025687186533, DECIMAL_4)