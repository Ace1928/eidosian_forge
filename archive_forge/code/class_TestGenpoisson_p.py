import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
from scipy.stats import poisson, nbinom
from statsmodels.tools.tools import Bunch
from statsmodels.distributions.discrete import (
class TestGenpoisson_p:

    def test_pmf_p1(self):
        poisson_pmf = poisson.pmf(1, 1)
        genpoisson_pmf = genpoisson_p.pmf(1, 1, 0, 1)
        assert_allclose(poisson_pmf, genpoisson_pmf, rtol=1e-15)

    def test_pmf_p2(self):
        poisson_pmf = poisson.pmf(2, 2)
        genpoisson_pmf = genpoisson_p.pmf(2, 2, 0, 2)
        assert_allclose(poisson_pmf, genpoisson_pmf, rtol=1e-15)

    def test_pmf_p5(self):
        poisson_pmf = poisson.pmf(10, 2)
        genpoisson_pmf_5 = genpoisson_p.pmf(10, 2, 1e-25, 5)
        assert_allclose(poisson_pmf, genpoisson_pmf_5, rtol=1e-12)

    def test_logpmf_p1(self):
        poisson_pmf = poisson.logpmf(5, 2)
        genpoisson_pmf = genpoisson_p.logpmf(5, 2, 0, 1)
        assert_allclose(poisson_pmf, genpoisson_pmf, rtol=1e-15)

    def test_logpmf_p2(self):
        poisson_pmf = poisson.logpmf(6, 1)
        genpoisson_pmf = genpoisson_p.logpmf(6, 1, 0, 2)
        assert_allclose(poisson_pmf, genpoisson_pmf, rtol=1e-15)