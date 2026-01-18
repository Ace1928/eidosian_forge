import io
import os
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import patsy
from statsmodels.api import families
from statsmodels.tools.sm_exceptions import (
from statsmodels.othermod.betareg import BetaModel
from .results import results_betareg as resultsb
class TestBetaMeth:

    @classmethod
    def setup_class(cls):
        formula = 'methylation ~ gender + CpG'
        mod = BetaModel.from_formula(formula, methylation, exog_precision_formula='~ age', link_precision=links.Log())
        cls.res1 = mod.fit(cov_type='eim')
        cls.res2 = resultsb.results_meth

    def test_basic(self):
        res1 = self.res1
        res2 = self.res2
        k_mean = 4
        p, se, zv, pv = res2.table_mean.T
        assert_allclose(res1.params[:k_mean], p, rtol=1e-06)
        assert_allclose(res1.bse[:k_mean], se, rtol=1e-06)
        assert_allclose(res1.tvalues[:k_mean], zv, rtol=1e-06)
        assert_allclose(res1.pvalues[:k_mean], pv, rtol=1e-05)
        p, se, zv, pv = res2.table_precision.T
        assert_allclose(res1.params[k_mean:], p, rtol=1e-06)
        assert_allclose(res1.bse[k_mean:], se, rtol=1e-06)
        assert_allclose(res1.tvalues[k_mean:], zv, rtol=1e-06)
        assert_allclose(res1.pvalues[k_mean:], pv, rtol=1e-05)
        assert_allclose(res1.llf, res2.loglik, rtol=1e-10)
        assert_allclose(res1.aic, res2.aic, rtol=1e-10)
        assert_allclose(res1.bic, res2.bic, rtol=1e-10)
        assert_allclose(res1.prsquared, res2.pseudo_r_squared, atol=0.01)
        assert_equal(res1.df_resid, res2.df_residual)
        assert_equal(res1.nobs, res2.nobs)
        df_c = res1.df_resid_null - res1.df_resid
        assert_equal(res1.k_null, 2)
        pv = 7.21872953868659e-18
        lln = 60.88809589492269
        llf = 104.14802840534323
        chisq = 86.51986502084107
        dfc = 4
        assert_equal(df_c, dfc)
        assert_allclose(res1.llf, llf, rtol=1e-10)
        assert_allclose(res1.llnull, lln, rtol=1e-10)
        assert_allclose(res1.llr, chisq, rtol=1e-10)
        assert_allclose(res1.llr_pvalue, pv, rtol=1e-06)

    def test_resid(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.fittedvalues, res2.resid['fittedvalues'], rtol=1e-08)
        assert_allclose(res1.resid, res2.resid['response'], atol=1e-08, rtol=1e-08)

    def test_oim(self):
        res1 = self.res1.model.fit()
        res2 = self.res2
        k_mean = 4
        p, se, zv, pv = res2.table_mean_oim.T
        assert_allclose(res1.params[:k_mean], p, rtol=1e-06)
        assert_allclose(res1.bse[:k_mean], se, rtol=1e-05)
        assert_allclose(res1.tvalues[:k_mean], zv, rtol=1e-05)
        assert_allclose(res1.pvalues[:k_mean], pv, atol=1e-06, rtol=1e-05)
        p, se, zv, pv = res2.table_precision_oim.T
        assert_allclose(res1.params[k_mean:], p, rtol=1e-06)
        assert_allclose(res1.bse[k_mean:], se, rtol=0.001)
        assert_allclose(res1.tvalues[k_mean:], zv, rtol=0.001)
        assert_allclose(res1.pvalues[k_mean:], pv, rtol=0.01)

    def test_predict_distribution(self):
        res1 = self.res1
        mean = res1.predict()
        var_ = res1.model._predict_var(res1.params)
        distr = res1.get_distribution()
        m2, v2 = distr.stats()
        assert_allclose(mean, m2, rtol=1e-13)
        assert_allclose(var_, v2, rtol=1e-13)
        var_r6 = [0.00031090848852102, 0.00024509604000073, 0.00037199753140565, 0.00028088261358738, 0.0002756111180035, 0.00033929220526847]
        n = 6
        assert_allclose(v2[:n], var_r6, rtol=1e-07)
        ex = res1.model.exog[:n]
        ex_prec = res1.model.exog_precision[:n]
        mean6 = res1.predict(ex, transform=False)
        prec = res1.predict(which='precision')
        prec6 = res1.predict(exog_precision=ex_prec, which='precision', transform=False)
        var6 = res1.model._predict_var(res1.params, exog=ex, exog_precision=ex_prec)
        assert_allclose(mean6, mean[:n], rtol=1e-13)
        assert_allclose(prec6, prec[:n], rtol=1e-13)
        assert_allclose(var6, var_[:n], rtol=1e-13)
        assert_allclose(var6, var_r6, rtol=1e-07)
        distr6 = res1.model.get_distribution(res1.params, exog=ex, exog_precision=ex_prec)
        m26, v26 = distr6.stats()
        assert_allclose(m26, m2[:n], rtol=1e-13)
        assert_allclose(v26, v2[:n], rtol=1e-13)
        distr6f = res1.get_distribution(exog=ex, exog_precision=ex_prec, transform=False)
        m26, v26 = distr6f.stats()
        assert_allclose(m26, m2[:n], rtol=1e-13)
        assert_allclose(v26, v2[:n], rtol=1e-13)
        df6 = methylation.iloc[:6]
        mean6f = res1.predict(df6)
        assert_allclose(mean6f, mean[:n], rtol=1e-13)
        distr6f = res1.get_distribution(exog=df6, exog_precision=ex_prec)
        m26, v26 = distr6f.stats()
        assert_allclose(m26, m2[:n], rtol=1e-13)
        assert_allclose(v26, v2[:n], rtol=1e-13)
        assert isinstance(distr6f.args[0], np.ndarray)
        pma = res1.get_prediction(which='mean', average=True)
        dfma = pma.summary_frame()
        assert_allclose(pma.predicted, mean.mean(), rtol=1e-13)
        assert_equal(dfma.shape, (1, 4))
        pm = res1.get_prediction(exog=df6, which='mean', average=False)
        dfm = pm.summary_frame()
        assert_allclose(pm.predicted, mean6, rtol=1e-13)
        assert_equal(dfm.shape, (6, 4))
        pv = res1.get_prediction(exog=df6, exog_precision=ex_prec, which='var', average=False)
        dfv = pv.summary_frame()
        assert_allclose(pv.predicted, var6, rtol=1e-13)
        assert_equal(dfv.shape, (6, 4))
        res1.get_prediction(which='linear', average=False)
        res1.get_prediction(which='precision', average=True)
        res1.get_prediction(exog_precision=ex_prec, which='precision', average=False)
        res1.get_prediction(which='linear-precision', average=True)
        pm = res1.get_prediction(exog=df6, which='mean', average=True)
        dfm = pm.summary_frame()
        aw = np.zeros(len(res1.model.endog))
        aw[:6] = 1
        aw /= aw.mean()
        pm6 = res1.get_prediction(exog=df6, which='mean', average=True)
        dfm6 = pm6.summary_frame()
        pmw = res1.get_prediction(which='mean', average=True, agg_weights=aw)
        dfmw = pmw.summary_frame()
        assert_allclose(pmw.predicted, pm6.predicted, rtol=1e-13)
        assert_allclose(dfmw, dfm6, rtol=1e-13)