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
class CheckRegressionResults:
    """
    res2 contains results from Rmodelwrap or were obtained from a statistical
    packages such as R, Stata, or SAS and were written to model_results
    """
    decimal_params = DECIMAL_4

    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, self.decimal_params)
    decimal_standarderrors = DECIMAL_4

    def test_standarderrors(self):
        assert_allclose(self.res1.bse, self.res2.bse, self.decimal_standarderrors)
    decimal_confidenceintervals = DECIMAL_4

    def test_confidenceintervals(self):
        conf1 = self.res1.conf_int()
        conf2 = self.res2.conf_int()
        for i in range(len(conf1)):
            assert_allclose(conf1[i][0], conf2[i][0], rtol=10 ** (-self.decimal_confidenceintervals))
            assert_allclose(conf1[i][1], conf2[i][1], rtol=10 ** (-self.decimal_confidenceintervals))
    decimal_conf_int_subset = DECIMAL_4

    def test_conf_int_subset(self):
        if len(self.res1.params) > 1:
            with pytest.warns(FutureWarning, match='cols is'):
                ci1 = self.res1.conf_int(cols=(1, 2))
            ci2 = self.res1.conf_int()[1:3]
            assert_almost_equal(ci1, ci2, self.decimal_conf_int_subset)
        else:
            pass
    decimal_scale = DECIMAL_4

    def test_scale(self):
        assert_almost_equal(self.res1.scale, self.res2.scale, self.decimal_scale)
    decimal_rsquared = DECIMAL_4

    def test_rsquared(self):
        assert_almost_equal(self.res1.rsquared, self.res2.rsquared, self.decimal_rsquared)
    decimal_rsquared_adj = DECIMAL_4

    def test_rsquared_adj(self):
        assert_almost_equal(self.res1.rsquared_adj, self.res2.rsquared_adj, self.decimal_rsquared_adj)

    def test_degrees(self):
        assert_equal(self.res1.model.df_model, self.res2.df_model)
        assert_equal(self.res1.model.df_resid, self.res2.df_resid)
    decimal_ess = DECIMAL_4

    def test_ess(self):
        assert_almost_equal(self.res1.ess, self.res2.ess, self.decimal_ess)
    decimal_ssr = DECIMAL_4

    def test_sumof_squaredresids(self):
        assert_almost_equal(self.res1.ssr, self.res2.ssr, self.decimal_ssr)
    decimal_mse_resid = DECIMAL_4

    def test_mse_resid(self):
        assert_almost_equal(self.res1.mse_model, self.res2.mse_model, self.decimal_mse_resid)
    decimal_mse_model = DECIMAL_4

    def test_mse_model(self):
        assert_almost_equal(self.res1.mse_resid, self.res2.mse_resid, self.decimal_mse_model)
    decimal_mse_total = DECIMAL_4

    def test_mse_total(self):
        assert_almost_equal(self.res1.mse_total, self.res2.mse_total, self.decimal_mse_total, err_msg='Test class %s' % self)
    decimal_fvalue = DECIMAL_4

    def test_fvalue(self):
        assert_almost_equal(self.res1.fvalue, self.res2.fvalue, self.decimal_fvalue)
    decimal_loglike = DECIMAL_4

    def test_loglike(self):
        assert_almost_equal(self.res1.llf, self.res2.llf, self.decimal_loglike)
    decimal_aic = DECIMAL_4

    def test_aic(self):
        assert_almost_equal(self.res1.aic, self.res2.aic, self.decimal_aic)
        aicc1 = self.res1.info_criteria('aicc')
        k = self.res1.df_model + self.res1.model.k_constant
        nobs = self.res1.model.nobs
        aicc2 = self.res1.aic + 2 * (k ** 2 + k) / (nobs - k - 1)
        assert_allclose(aicc1, aicc2, rtol=1e-10)
        hqic1 = self.res1.info_criteria('hqic')
        hqic2 = self.res1.aic - 2 * k + 2 * np.log(np.log(nobs)) * k
        assert_allclose(hqic1, hqic2, rtol=1e-10)
    decimal_bic = DECIMAL_4

    def test_bic(self):
        assert_almost_equal(self.res1.bic, self.res2.bic, self.decimal_bic)
    decimal_pvalues = DECIMAL_4

    def test_pvalues(self):
        assert_almost_equal(self.res1.pvalues, self.res2.pvalues, self.decimal_pvalues)
    decimal_wresid = DECIMAL_4

    def test_wresid(self):
        assert_almost_equal(self.res1.wresid, self.res2.wresid, self.decimal_wresid)
    decimal_resids = DECIMAL_4

    def test_resids(self):
        assert_almost_equal(self.res1.resid, self.res2.resid, self.decimal_resids)
    decimal_norm_resids = DECIMAL_4

    def test_norm_resids(self):
        assert_almost_equal(self.res1.resid_pearson, self.res2.resid_pearson, self.decimal_norm_resids)