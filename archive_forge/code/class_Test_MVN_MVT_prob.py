import numpy as np
from numpy.testing import assert_almost_equal,  assert_allclose
from statsmodels.sandbox.distributions.multivariate import (
from statsmodels.sandbox.distributions.mv_normal import MVT, MVNormal
class Test_MVN_MVT_prob:

    @classmethod
    def setup_class(cls):
        cls.corr_equal = np.asarray([[1.0, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])
        cls.a = -1 * np.ones(3)
        cls.b = 3 * np.ones(3)
        cls.df = 4
        corr2 = cls.corr_equal.copy()
        corr2[2, 1] = -0.5
        cls.corr2 = corr2

    def test_mvn_mvt_1(self):
        a, b = (self.a, self.b)
        df = self.df
        corr_equal = self.corr_equal
        probmvt_R = 0.60414
        probmvn_R = 0.67397
        assert_almost_equal(probmvt_R, mvstdtprob(a, b, corr_equal, df), 4)
        assert_almost_equal(probmvn_R, mvstdnormcdf(a, b, corr_equal, abseps=1e-05), 4)
        mvn_high = mvstdnormcdf(a, b, corr_equal, abseps=1e-08, maxpts=10000000)
        assert_almost_equal(probmvn_R, mvn_high, 5)

    def test_mvn_mvt_2(self):
        a, b = (self.a, self.b)
        df = self.df
        corr2 = self.corr2
        probmvn_R = 0.6472497
        probmvt_R = 0.5881863
        assert_almost_equal(probmvt_R, mvstdtprob(a, b, corr2, df), 4)
        assert_almost_equal(probmvn_R, mvstdnormcdf(a, b, corr2, abseps=1e-05), 4)

    def test_mvn_mvt_3(self):
        a, b = (self.a, self.b)
        df = self.df
        corr2 = self.corr2
        a2 = a.copy()
        a2[:] = -np.inf
        probmvn_R = 0.9961141
        probmvt_R = 0.9522146
        quadkwds = {'epsabs': 1e-08}
        probmvt = mvstdtprob(a2, b, corr2, df, quadkwds=quadkwds)
        assert_allclose(probmvt_R, probmvt, atol=0.0005)
        probmvn = mvstdnormcdf(a2, b, corr2, maxpts=100000, abseps=1e-05)
        assert_allclose(probmvn_R, probmvn, atol=0.0001)

    def test_mvn_mvt_4(self):
        a, bl = (self.a, self.b)
        df = self.df
        corr2 = self.corr2
        a2 = a.copy()
        a2[:] = -np.inf
        probmvn_R = 0.1666667
        probmvt_R = 0.1666667
        assert_almost_equal(probmvt_R, mvstdtprob(np.zeros(3), -a2, corr2, df), 4)
        assert_almost_equal(probmvn_R, mvstdnormcdf(np.zeros(3), -a2, corr2, maxpts=100000, abseps=1e-05), 4)

    def test_mvn_mvt_5(self):
        a, bl = (self.a, self.b)
        df = self.df
        corr2 = self.corr2
        a3 = np.array([0.5, -0.5, 0.5])
        probmvn_R = 0.06910487
        probmvt_R = 0.05797867
        assert_almost_equal(mvstdtprob(a3, a3 + 1, corr2, df), probmvt_R, 4)
        assert_almost_equal(probmvn_R, mvstdnormcdf(a3, a3 + 1, corr2, maxpts=100000, abseps=1e-05), 4)