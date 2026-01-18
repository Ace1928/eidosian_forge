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
class TestFTest2:
    """
    A joint test that the coefficient on
    GNP = the coefficient on UNEMP  and that the coefficient on
    POP = the coefficient on YEAR for the Longley dataset.

    Ftest1 is from statsmodels.  Results are from Rpy using R's car library.
    """

    @classmethod
    def setup_class(cls):
        data = longley.load()
        columns = [f'x{i}' for i in range(1, data.exog.shape[1] + 1)]
        data.exog.columns = columns
        data.exog = add_constant(data.exog, prepend=False)
        res1 = OLS(data.endog, data.exog).fit()
        R2 = [[0, 1, -1, 0, 0, 0, 0], [0, 0, 0, 0, 1, -1, 0]]
        cls.Ftest1 = res1.f_test(R2)
        hyp = 'x2 = x3, x5 = x6'
        cls.NewFtest1 = res1.f_test(hyp)

    def test_new_ftest(self):
        assert_equal(self.NewFtest1.fvalue, self.Ftest1.fvalue)

    def test_fvalue(self):
        assert_almost_equal(self.Ftest1.fvalue, 9.74046187329682, DECIMAL_4)

    def test_pvalue(self):
        assert_almost_equal(self.Ftest1.pvalue, 0.005605288531749346, DECIMAL_4)

    def test_df_denom(self):
        assert_equal(self.Ftest1.df_denom, 9)

    def test_df_num(self):
        assert_equal(self.Ftest1.df_num, 2)