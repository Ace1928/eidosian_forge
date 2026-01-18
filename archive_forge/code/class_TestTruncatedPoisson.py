import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
from scipy.stats import poisson, nbinom
from statsmodels.tools.tools import Bunch
from statsmodels.distributions.discrete import (
class TestTruncatedPoisson:
    """
    Test Truncated Poisson distribution
    """

    def test_pmf_zero(self):
        poisson_pmf = poisson.pmf(2, 2) / poisson.sf(0, 2)
        tpoisson_pmf = truncatedpoisson.pmf(2, 2, 0)
        assert_allclose(poisson_pmf, tpoisson_pmf, rtol=1e-07)

    def test_logpmf_zero(self):
        poisson_logpmf = poisson.logpmf(2, 2) - np.log(poisson.sf(0, 2))
        tpoisson_logpmf = truncatedpoisson.logpmf(2, 2, 0)
        assert_allclose(poisson_logpmf, tpoisson_logpmf, rtol=1e-07)

    def test_pmf(self):
        poisson_pmf = poisson.pmf(4, 6) / (1 - poisson.cdf(2, 6))
        tpoisson_pmf = truncatedpoisson.pmf(4, 6, 2)
        assert_allclose(poisson_pmf, tpoisson_pmf, rtol=1e-07)

    def test_logpmf(self):
        poisson_logpmf = poisson.logpmf(4, 6) - np.log(poisson.sf(2, 6))
        tpoisson_logpmf = truncatedpoisson.logpmf(4, 6, 2)
        assert_allclose(poisson_logpmf, tpoisson_logpmf, rtol=1e-07)