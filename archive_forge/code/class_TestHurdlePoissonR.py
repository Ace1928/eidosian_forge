import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from statsmodels import datasets
from statsmodels.tools.tools import add_constant
from statsmodels.tools.testing import Holder
from statsmodels.tools.sm_exceptions import (
from statsmodels.distributions.discrete import (
from statsmodels.discrete.truncated_model import (
from statsmodels.sandbox.regression.tests.test_gmm_poisson import DATA
from .results.results_discrete import RandHIE
from .results import results_truncated as results_t
from .results import results_truncated_st as results_ts
class TestHurdlePoissonR:

    @classmethod
    def setup_class(cls):
        endog = DATA['docvis']
        exog_names = ['const', 'aget', 'totchr']
        exog = DATA[exog_names]
        cls.res1 = HurdleCountModel(endog, exog).fit(method='newton', maxiter=300)
        cls.res2 = results_t.hurdle_poisson

    def test_basic(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.llf, res2.loglik, rtol=1e-08)
        pt2 = res2.params_table
        assert_allclose(res1.params, pt2[:, 0], atol=1e-05)
        assert_allclose(res1.bse, pt2[:, 1], atol=1e-05)
        assert_allclose(res1.tvalues, pt2[:, 2], rtol=0.0005, atol=0.0005)
        assert_allclose(res1.pvalues, pt2[:, 3], rtol=0.0005, atol=1e-07)
        assert_equal(res1.df_resid, res2.df_residual)
        assert_equal(res1.df_model, res2.df_null - res2.df_residual)
        assert_allclose(res1.aic, res2.aic, rtol=1e-08)
        idx = np.concatenate((np.arange(3, 6), np.arange(3)))
        vcov = res2.vcov[idx[:, None], idx]
        assert_allclose(np.asarray(res1.cov_params()), vcov, rtol=0.0001, atol=1e-08)

    def test_predict(self):
        res1 = self.res1
        res2 = self.res2
        ex = res1.model.exog.mean(0, keepdims=True)
        mu1 = res1.results_zero.predict(ex)
        prob_zero = np.exp(-mu1)
        prob_nz = 1 - prob_zero
        assert_allclose(prob_nz, res2.predict_zero, rtol=0.0005, atol=0.0005)
        prob_nz_ = res1.results_zero.model._prob_nonzero(mu1, res1.params[:4])
        assert_allclose(prob_nz_, res2.predict_zero, rtol=0.0005, atol=0.0005)
        mean_main = res1.results_count.predict(ex, which='mean-main')
        assert_allclose(mean_main, res2.predict_mean_main, rtol=0.0005, atol=0.0005)
        prob_main = res1.results_count.predict(ex, which='prob')[0] * prob_nz
        prob_main[0] = np.squeeze(prob_zero)
        assert_allclose(prob_main[:4], res2.predict_prob, rtol=0.0005, atol=0.0005)
        assert_allclose(mean_main * prob_nz, res2.predict_mean, rtol=0.001, atol=0.0005)
        m = res1.predict(ex)
        assert_allclose(m, res2.predict_mean, rtol=1e-06, atol=5e-07)
        mm = res1.predict(ex, which='mean-main')
        assert_allclose(mm, res2.predict_mean_main, rtol=1e-07, atol=1e-07)
        mnz = res1.predict(ex, which='mean-nonzero')
        assert_allclose(mnz, res2.predict_mean / (1 - res2.predict_prob[0]), rtol=5e-07, atol=5e-07)
        prob_main = res1.predict(ex, which='prob-main')
        pt = res1.predict(ex, which='prob-trunc')
        assert_allclose(prob_main / (1 - pt), res2.predict_zero, rtol=0.0005, atol=0.0005)
        probs = res1.predict(ex, which='prob')[0]
        assert_allclose(probs[:4], res2.predict_prob, rtol=1e-05, atol=1e-06)
        k_ex = 5
        ex5 = res1.model.exog[:k_ex]
        p1a = res1.predict(ex5, which='prob', y_values=np.arange(3))
        p1b = res1.get_prediction(ex5, which='prob', y_values=np.arange(3))
        assert_allclose(p1a, p1b.predicted, rtol=1e-10, atol=1e-10)
        p2a = res1.predict(which='prob', y_values=np.arange(3))
        p2b = res1.get_prediction(which='prob', y_values=np.arange(3), average=True)
        assert_allclose(p2a.mean(0), p2b.predicted, rtol=1e-10, atol=1e-10)
        for which in ['mean', 'mean-main', 'prob-main', 'prob-zero', 'linear']:
            p3a = res1.predict(ex5, which=which)
            p3b = res1.get_prediction(ex5, which=which)
            assert_allclose(p3a, p3b.predicted, rtol=1e-10, atol=1e-10)
            assert p3b.summary_frame().shape == (k_ex, 4)
        resid_p1 = res1.resid_pearson[:5]
        resid_p2 = np.asarray([-1.5892397298897, -0.3239276467705, -1.5878941800178, 0.6613236544236, -0.6690997162962])
        assert_allclose(resid_p1, resid_p2, rtol=1e-05, atol=1e-05)