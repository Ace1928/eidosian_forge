from statsmodels.compat.python import lmap
import numpy as np
import pandas
from scipy import stats
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.sandbox.regression import gmm
from numpy.testing import assert_allclose, assert_equal
class CheckGMM:
    params_tol = [5e-06, 5e-06]
    bse_tol = [5e-07, 5e-07]
    q_tol = [5e-06, 1e-09]
    j_tol = [5e-05, 1e-09]

    def test_basic(self):
        res1, res2 = (self.res1, self.res2)
        rtol, atol = self.params_tol
        assert_allclose(res1.params, res2.params, rtol=rtol, atol=0)
        assert_allclose(res1.params, res2.params, rtol=0, atol=atol)
        rtol, atol = self.bse_tol
        assert_allclose(res1.bse, res2.bse, rtol=rtol, atol=0)
        assert_allclose(res1.bse, res2.bse, rtol=0, atol=atol)

    def test_other(self):
        res1, res2 = (self.res1, self.res2)
        rtol, atol = self.q_tol
        assert_allclose(res1.q, res2.Q, rtol=atol, atol=rtol)
        rtol, atol = self.j_tol
        assert_allclose(res1.jval, res2.J, rtol=atol, atol=rtol)
        j, jpval, jdf = res1.jtest()
        assert_allclose(res1.jval, res2.J, rtol=13, atol=13)
        pval = stats.chi2.sf(res2.J, res2.J_df)
        assert_allclose(jpval, pval, rtol=rtol, atol=atol)
        assert_equal(jdf, res2.J_df)

    @pytest.mark.smoke
    def test_summary(self):
        res1 = self.res1
        summ = res1.summary()
        assert_equal(len(summ.tables[1]), len(res1.params) + 1)