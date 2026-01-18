import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
from scipy.stats import poisson, nbinom
from statsmodels.tools.tools import Bunch
from statsmodels.distributions.discrete import (
class CheckDiscretized:

    def convert_params(self, params):
        args = params.tolist()
        args.insert(-1, 0)
        return args

    def test_basic(self):
        d_offset = self.d_offset
        ddistr = self.ddistr
        paramg = self.paramg
        paramd = self.paramd
        shapes = self.shapes
        start_params = self.start_params
        np.random.seed(987146)
        dp = DiscretizedCount(ddistr, d_offset)
        assert dp.shapes == shapes
        xi = np.arange(5)
        p = dp._pmf(xi, *paramd)
        cdf1 = ddistr.cdf(xi, *paramg)
        p1 = np.diff(cdf1)
        assert_allclose(p[:len(p1)], p1, rtol=1e-13)
        cdf = dp._cdf(xi, *paramd)
        assert_allclose(cdf[:len(cdf1) - 1], cdf1[1:], rtol=1e-13)
        p2 = dp.pmf(xi, *paramd)
        assert_allclose(p2, p, rtol=1e-13)
        cdf2 = dp.cdf(xi, *paramd)
        assert_allclose(cdf2, cdf, rtol=1e-13)
        sf = dp.sf(xi, *paramd)
        assert_allclose(sf, 1 - cdf, rtol=1e-13)
        nobs = 2000
        xx = dp.rvs(*paramd, size=nobs)
        assert len(xx) == nobs
        assert xx.var() > 0.001
        mod = DiscretizedModel(xx, distr=dp)
        res = mod.fit(start_params=start_params)
        p = mod.predict(res.params, which='probs')
        args = self.convert_params(res.params)
        p1 = -np.diff(ddistr.sf(np.arange(21), *args))
        assert_allclose(p, p1, rtol=1e-13)
        p1 = np.diff(ddistr.cdf(np.arange(21), *args))
        assert_allclose(p, p1, rtol=1e-13, atol=1e-15)
        freq = np.bincount(xx.astype(int))
        k = len(freq)
        if k > 10:
            k = 10
            freq[k - 1] += freq[k:].sum()
            freq = freq[:k]
        p = mod.predict(res.params, which='probs', k_max=k)
        p[k - 1] += 1 - p[:k].sum()
        tchi2 = stats.chisquare(freq, p[:k] * nobs)
        assert tchi2.pvalue > 0.01
        dfr = mod.get_distr(res.params)
        nobs_rvs = 500
        rvs = dfr.rvs(size=nobs_rvs)
        freq = np.bincount(rvs)
        p = mod.predict(res.params, which='probs', k_max=nobs_rvs)
        k = len(freq)
        p[k - 1] += 1 - p[:k].sum()
        tchi2 = stats.chisquare(freq, p[:k] * nobs_rvs)
        assert tchi2.pvalue > 0.01
        q = dfr.ppf(dfr.cdf(np.arange(-1, 5) + 1e-06))
        q1 = np.array([-1.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        assert_equal(q, q1)
        p = np.maximum(dfr.cdf(np.arange(-1, 5)) - 1e-06, 0)
        q = dfr.ppf(p)
        q1 = np.arange(-1, 5)
        assert_equal(q, q1)
        q = dfr.ppf(dfr.cdf(np.arange(5)))
        q1 = np.arange(0, 5)
        assert_equal(q, q1)
        q = dfr.isf(1 - dfr.cdf(np.arange(-1, 5) + 1e-06))
        q1 = np.array([-1.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        assert_equal(q, q1)