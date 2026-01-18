import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels.tools.tools import add_constant
from statsmodels.base._prediction_inference import PredictionResultsMonotonic
from statsmodels.discrete.discrete_model import (
from statsmodels.discrete.count_model import (
from statsmodels.sandbox.regression.tests.test_gmm_poisson import DATA
from .results import results_predict as resp
class CheckPredict:

    def test_basic(self):
        res1 = self.res1
        res2 = self.res2
        sl1 = slice(self.k_infl, -1, None)
        sl2 = slice(0, -(self.k_infl + 1), None)
        assert_allclose(res1.params[sl1], res2.params[sl2], rtol=self.rtol)
        assert_allclose(res1.bse[sl1], res2.bse[sl2], rtol=30 * self.rtol)
        params1 = np.asarray(res1.params)
        params2 = np.asarray(res2.params)
        assert_allclose(params1[-1], np.exp(params2[-1]), rtol=self.rtol)

    def test_predict(self):
        res1 = self.res1
        res2 = self.res2
        ex = np.asarray(exog).mean(0)
        rdf = res2.results_margins_atmeans
        pred = res1.get_prediction(ex, **self.pred_kwds_mean)
        assert_allclose(pred.predicted, rdf['b'].iloc[0], rtol=0.0001)
        assert_allclose(pred.se, rdf['se'].iloc[0], rtol=0.0001, atol=0.0001)
        if isinstance(pred, PredictionResultsMonotonic):
            ci = pred.conf_int()[0]
            assert_allclose(ci[0], rdf['ll'].iloc[0], rtol=0.001, atol=0.0001)
            assert_allclose(ci[1], rdf['ul'].iloc[0], rtol=0.001, atol=0.0001)
            ci = pred.conf_int(method='delta')[0]
            assert_allclose(ci[0], rdf['ll'].iloc[0], rtol=0.0001, atol=0.0001)
            assert_allclose(ci[1], rdf['ul'].iloc[0], rtol=0.0001, atol=0.0001)
        else:
            ci = pred.conf_int()[0]
            assert_allclose(ci[0], rdf['ll'].iloc[0], rtol=0.0001, atol=0.0001)
            assert_allclose(ci[1], rdf['ul'].iloc[0], rtol=0.0001, atol=0.0001)
        stat, _ = pred.t_test()
        assert_allclose(stat, pred.tvalues, rtol=0.0001, atol=0.0001)
        rdf = res2.results_margins_mean
        pred = res1.get_prediction(average=True, **self.pred_kwds_mean)
        assert_allclose(pred.predicted, rdf['b'].iloc[0], rtol=0.0003)
        assert_allclose(pred.se, rdf['se'].iloc[0], rtol=0.003, atol=0.0001)
        if isinstance(pred, PredictionResultsMonotonic):
            ci = pred.conf_int()[0]
            assert_allclose(ci[0], rdf['ll'].iloc[0], rtol=0.001, atol=0.0001)
            assert_allclose(ci[1], rdf['ul'].iloc[0], rtol=0.001, atol=0.0001)
            ci = pred.conf_int(method='delta')[0]
            assert_allclose(ci[0], rdf['ll'].iloc[0], rtol=0.0001, atol=0.0001)
            assert_allclose(ci[1], rdf['ul'].iloc[0], rtol=0.0001, atol=0.0001)
        else:
            ci = pred.conf_int()[0]
            assert_allclose(ci[0], rdf['ll'].iloc[0], rtol=0.0005, atol=0.0001)
            assert_allclose(ci[1], rdf['ul'].iloc[0], rtol=0.0005, atol=0.0001)
        stat, _ = pred.t_test()
        assert_allclose(stat, pred.tvalues, rtol=0.0001, atol=0.0001)
        rdf = res2.results_margins_atmeans
        pred = res1.get_prediction(ex, which='prob', y_values=np.arange(2), **self.pred_kwds_mean)
        assert_allclose(pred.predicted, rdf['b'].iloc[1:3], rtol=0.0003)
        assert_allclose(pred.se, rdf['se'].iloc[1:3], rtol=0.003, atol=0.0001)
        ci = pred.conf_int()
        assert_allclose(ci[:, 0], rdf['ll'].iloc[1:3], rtol=0.0005, atol=0.0001)
        assert_allclose(ci[:, 1], rdf['ul'].iloc[1:3], rtol=0.0005, atol=0.0001)
        stat, _ = pred.t_test()
        assert_allclose(stat, pred.tvalues, rtol=0.0001, atol=0.0001)
        rdf = res2.results_margins_mean
        pred = res1.get_prediction(which='prob', y_values=np.arange(2), average=True, **self.pred_kwds_mean)
        assert_allclose(pred.predicted, rdf['b'].iloc[1:3], rtol=0.005)
        assert_allclose(pred.se, rdf['se'].iloc[1:3], rtol=0.003, atol=0.0005)
        ci = pred.conf_int()
        assert_allclose(ci[:, 0], rdf['ll'].iloc[1:3], rtol=0.0005, atol=0.001)
        assert_allclose(ci[:, 1], rdf['ul'].iloc[1:3], rtol=0.0005, atol=0.005)
        stat, _ = pred.t_test()
        assert_allclose(stat, pred.tvalues, rtol=0.0001, atol=0.0001)
        stat, _ = pred.t_test(value=pred.predicted)
        assert_equal(stat, 0)
        df6 = exog[:6]
        aw = np.zeros(len(res1.model.endog))
        aw[:6] = 1
        aw /= aw.mean()
        pm6 = res1.get_prediction(exog=df6, which='mean', average=True, **self.pred_kwds_6)
        dfm6 = pm6.summary_frame()
        pmw = res1.get_prediction(which='mean', average=True, agg_weights=aw)
        dfmw = pmw.summary_frame()
        assert_allclose(pmw.predicted, pm6.predicted, rtol=1e-13)
        assert_allclose(dfmw, dfm6, rtol=1e-07)

    def test_diagnostic(self):
        res1 = self.res1
        dia = res1.get_diagnostic(y_max=21)
        res_chi2 = dia.test_chisquare_prob(bin_edges=np.arange(4))
        assert_equal(res_chi2.diff1.shape[1], 3)
        assert_equal(dia.probs_predicted.shape[1], 22)
        try:
            dia.plot_probs(upp_xlim=20)
        except ImportError:
            pass