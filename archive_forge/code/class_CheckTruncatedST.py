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
class CheckTruncatedST:

    def test_basic(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.llf, res2.ll, rtol=1e-08)
        assert_allclose(res1.llnull, res2.ll_0, rtol=5e-06)
        pt2 = res2.params_table
        k = res1.model.exog.shape[1]
        assert_allclose(res1.params[:k], res2.params[:k], atol=1e-05)
        assert_allclose(res1.bse[:k], pt2[:k, 1], atol=1e-05)
        assert_allclose(res1.tvalues[:k], pt2[:k, 2], rtol=0.0005, atol=0.0005)
        assert_allclose(res1.pvalues[:k], pt2[:k, 3], rtol=0.0005, atol=1e-07)
        assert_equal(res1.df_model, res2.df_m)
        assert_allclose(res1.aic, res2.icr[-2], rtol=1e-08)
        assert_allclose(res1.bic, res2.icr[-1], rtol=1e-08)
        nobs = res1.model.endog.shape[0]
        assert_equal((res1.model.endog < 1).sum(), 0)
        assert_equal(res1.df_resid, nobs - len(res1.params))

    def test_predict(self):
        res1 = self.res1
        res2 = self.res2
        rdf = res2.margins_means.table
        pred = res1.get_prediction(which='mean-main', average=True)
        assert_allclose(pred.predicted, rdf[0], rtol=5e-05)
        assert_allclose(pred.se, rdf[1], rtol=0.0005, atol=1e-10)
        ci = pred.conf_int()[0]
        assert_allclose(ci[0], rdf[4], rtol=1e-05, atol=1e-10)
        assert_allclose(ci[1], rdf[5], rtol=1e-05, atol=1e-10)
        ex = res1.model.exog.mean(0)
        rdf = res2.margins_atmeans.table
        pred = res1.get_prediction(ex, which='mean-main')
        assert_allclose(pred.predicted, rdf[0], rtol=5e-05)
        assert_allclose(pred.se, rdf[1], rtol=0.0005, atol=1e-10)
        ci = pred.conf_int()[0]
        assert_allclose(ci[0], rdf[4], rtol=5e-05, atol=1e-10)
        assert_allclose(ci[1], rdf[5], rtol=5e-05, atol=1e-10)
        rdf = res2.margins_cm.table
        try:
            pred = res1.get_prediction(average=True)
        except NotImplementedError:
            pred = None
        if pred is not None:
            assert_allclose(pred.predicted, rdf[0], rtol=5e-05)
            assert_allclose(pred.se, rdf[1], rtol=1e-05, atol=1e-10)
            ci = pred.conf_int()[0]
            assert_allclose(ci[0], rdf[4], rtol=1e-05, atol=1e-10)
            assert_allclose(ci[1], rdf[5], rtol=1e-05, atol=1e-10)
        ex = res1.model.exog.mean(0)
        rdf = res2.margins_cpr.table
        start_idx = res1.model.truncation + 1
        k = rdf.shape[0] + res1.model.truncation
        pred = res1.get_prediction(which='prob', average=True)
        assert_allclose(pred.predicted[start_idx:k], rdf[:-1, 0], rtol=5e-05)
        assert_allclose(pred.se[start_idx:k], rdf[:-1, 1], rtol=0.0005, atol=1e-10)
        ci = pred.conf_int()[start_idx:k]
        assert_allclose(ci[:, 0], rdf[:-1, 4], rtol=5e-05, atol=1e-10)
        assert_allclose(ci[:, 1], rdf[:-1, 5], rtol=5e-05, atol=1e-10)
        ex = res1.model.exog.mean(0)
        rdf = res2.margins_pr.table
        k = rdf.shape[0] - 1
        pred = res1.get_prediction(which='prob-base', average=True)
        assert_allclose(pred.predicted[:k], rdf[:-1, 0], rtol=5e-05)
        assert_allclose(pred.se[:k], rdf[:-1, 1], rtol=0.0008, atol=1e-10)
        ci = pred.conf_int()[:k]
        assert_allclose(ci[:, 0], rdf[:-1, 4], rtol=0.0005, atol=1e-10)
        assert_allclose(ci[:, 1], rdf[:-1, 5], rtol=0.0005, atol=1e-10)