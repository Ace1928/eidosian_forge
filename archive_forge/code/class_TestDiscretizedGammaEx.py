import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
from scipy.stats import poisson, nbinom
from statsmodels.tools.tools import Bunch
from statsmodels.distributions.discrete import (
class TestDiscretizedGammaEx:

    def test_all(self):
        freq = [46, 76, 24, 9, 1]
        y = np.repeat(np.arange(5), freq)
        res1 = Bunch(params=[3.52636, 0.425617], llf=-187.469, chi2=1.701208, df_model=0, p=0.4272, aic=378.938, probs=[46.48, 73.72, 27.88, 6.5, 1.42])
        dp = DiscretizedCount(stats.gamma)
        mod = DiscretizedModel(y, distr=dp)
        res = mod.fit(start_params=[1, 1])
        nobs = len(y)
        assert_allclose(res.params, res1.params, rtol=1e-05)
        assert_allclose(res.llf, res1.llf, atol=0.006)
        assert_allclose(res.aic, res1.aic, atol=0.006)
        assert_equal(res.df_model, res1.df_model)
        probs = mod.predict(res.params, which='probs')
        probs_trunc = probs[:len(res1.probs)]
        probs_trunc[-1] += 1 - probs_trunc.sum()
        assert_allclose(probs_trunc * nobs, res1.probs, atol=0.06)
        assert_allclose(np.sum(freq), (probs_trunc * nobs).sum(), rtol=1e-10)
        res_chi2 = stats.chisquare(freq, probs_trunc * nobs, ddof=len(res.params))
        assert_allclose(res_chi2.statistic, 1.70409356, rtol=1e-07)
        assert_allclose(res_chi2.pvalue, 0.426541, rtol=1e-07)
        res.summary()
        np.random.seed(987146)
        res_boots = res.bootstrap()
        assert_allclose(res.params, res_boots[0], rtol=0.05)
        assert_allclose(res.bse, res_boots[1], rtol=0.05)