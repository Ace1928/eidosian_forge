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
class TestGLS_large_data(TestDataDimensions):

    @classmethod
    def setup_class(cls):
        super().setup_class()
        nobs = 1000
        y = np.random.randn(nobs, 1)
        x = np.random.randn(nobs, 20)
        sigma = np.ones_like(y)
        cls.gls_res = GLS(y, x, sigma=sigma).fit()
        cls.gls_res_scalar = GLS(y, x, sigma=1).fit()
        cls.gls_res_none = GLS(y, x).fit()
        cls.ols_res = OLS(y, x).fit()

    def test_large_equal_params(self):
        assert_almost_equal(self.ols_res.params, self.gls_res.params, DECIMAL_7)

    def test_large_equal_loglike(self):
        assert_almost_equal(self.ols_res.llf, self.gls_res.llf, DECIMAL_7)

    def test_large_equal_params_none(self):
        assert_almost_equal(self.gls_res.params, self.gls_res_none.params, DECIMAL_7)