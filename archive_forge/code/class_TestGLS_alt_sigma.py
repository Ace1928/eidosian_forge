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
class TestGLS_alt_sigma(CheckRegressionResults):
    """
    Test that GLS with no argument is equivalent to OLS.
    """

    @classmethod
    def setup_class(cls):
        data = longley.load()
        endog = np.asarray(data.endog)
        exog = np.asarray(data.exog)
        exog = add_constant(exog, prepend=False)
        ols_res = OLS(endog, exog).fit()
        gls_res = GLS(endog, exog).fit()
        gls_res_scalar = GLS(endog, exog, sigma=1)
        cls.endog = endog
        cls.exog = exog
        cls.res1 = gls_res
        cls.res2 = ols_res
        cls.res3 = gls_res_scalar

    def test_wrong_size_sigma_1d(self):
        n = len(self.endog)
        assert_raises(ValueError, GLS, self.endog, self.exog, sigma=np.ones(n - 1))

    def test_wrong_size_sigma_2d(self):
        n = len(self.endog)
        assert_raises(ValueError, GLS, self.endog, self.exog, sigma=np.ones((n - 1, n - 1)))

    @pytest.mark.skip('Test does not raise but should')
    def test_singular_sigma(self):
        n = len(self.endog)
        sigma = np.ones((n, n)) + np.diag(np.ones(n))
        sigma[0, 1] = sigma[1, 0] = 2
        assert np.linalg.matrix_rank(sigma) == n - 1
        with pytest.raises(np.linalg.LinAlgError):
            GLS(self.endog, self.exog, sigma=sigma)