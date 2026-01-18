import numpy as np
from numpy.testing import assert_allclose, assert_equal  #noqa
from statsmodels.stats import weightstats
import statsmodels.stats.multivariate as smmv  # pytest cannot import test_xxx
from statsmodels.stats.multivariate import confint_mvmean_fromstats
from statsmodels.tools.testing import Holder
class TestCovStructure:

    @classmethod
    def setup_class(cls):
        cls.cov = np.array([[28.965925000000002, 17.215358333333327, 2.6945666666666654], [17.215358333333327, 21.452852666666672, 6.044527833333332], [2.6945666666666654, 6.044527833333332, 13.599042333333331]])
        cls.nobs = 25

    def test_spherical(self):
        cov, nobs = (self.cov, self.nobs)
        p_chi2 = 0.0006422366870356
        chi2 = 21.53275509455011
        stat, pv = smmv.test_cov_spherical(cov, nobs)
        assert_allclose(stat, chi2, rtol=1e-07)
        assert_allclose(pv, p_chi2, rtol=1e-06)

    def test_diagonal(self):
        cov, nobs = (self.cov, self.nobs)
        p_chi2 = 0.0004589987613319
        chi2 = 17.91025335733012
        stat, pv = smmv.test_cov_diagonal(cov, nobs)
        assert_allclose(stat, chi2, rtol=1e-08)
        assert_allclose(pv, p_chi2, rtol=1e-07)

    def test_blockdiagonal(self):
        cov, nobs = (self.cov, self.nobs)
        p_chi2 = 0.1721758850671037
        chi2 = 3.518477474111563
        block_len = [2, 1]
        stat, pv = smmv.test_cov_blockdiagonal(cov, nobs, block_len)
        assert_allclose(stat, chi2, rtol=1e-07)
        assert_allclose(pv, p_chi2, rtol=1e-06)

    def test_covmat(self):
        cov, nobs = (self.cov, self.nobs)
        p_chi2 = 0.4837049015162541
        chi2 = 5.481422374989864
        cov_null = np.array([[30, 15, 0], [15, 20, 0], [0, 0, 10]])
        stat, pv = smmv.test_cov(cov, nobs, cov_null)
        assert_allclose(stat, chi2, rtol=1e-07)
        assert_allclose(pv, p_chi2, rtol=1e-06)