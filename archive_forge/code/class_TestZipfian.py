import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
class TestZipfian:

    def test_zipfian_asymptotic(self):
        a = 6.5
        N = 10000000
        k = np.arange(1, 21)
        assert_allclose(zipfian.pmf(k, a, N), zipf.pmf(k, a))
        assert_allclose(zipfian.cdf(k, a, N), zipf.cdf(k, a))
        assert_allclose(zipfian.sf(k, a, N), zipf.sf(k, a))
        assert_allclose(zipfian.stats(a, N, moments='msvk'), zipf.stats(a, moments='msvk'))

    def test_zipfian_continuity(self):
        alt1, agt1 = (0.99999999, 1.00000001)
        N = 30
        k = np.arange(1, N + 1)
        assert_allclose(zipfian.pmf(k, alt1, N), zipfian.pmf(k, agt1, N), rtol=5e-07)
        assert_allclose(zipfian.cdf(k, alt1, N), zipfian.cdf(k, agt1, N), rtol=5e-07)
        assert_allclose(zipfian.sf(k, alt1, N), zipfian.sf(k, agt1, N), rtol=5e-07)
        assert_allclose(zipfian.stats(alt1, N, moments='msvk'), zipfian.stats(agt1, N, moments='msvk'), rtol=5e-07)

    def test_zipfian_R(self):
        np.random.seed(0)
        k = np.random.randint(1, 20, size=10)
        a = np.random.rand(10) * 10 + 1
        n = np.random.randint(1, 100, size=10)
        pmf = [0.008076972, 2.950214e-05, 0.9799333, 3.216601e-06, 0.0003158895, 3.412497e-05, 4.350472e-10, 2.405773e-06, 5.860662e-06, 0.0001053948]
        cdf = [0.8964133, 0.9998666, 0.9799333, 0.9999995, 0.9998584, 0.9999458, 1.0, 0.999992, 0.9999977, 0.9998498]
        assert_allclose(zipfian.pmf(k, a, n)[1:], pmf[1:], rtol=1e-06)
        assert_allclose(zipfian.cdf(k, a, n)[1:], cdf[1:], rtol=5e-05)
    np.random.seed(0)
    naive_tests = np.vstack((np.logspace(-2, 1, 10), np.random.randint(2, 40, 10))).T

    @pytest.mark.parametrize('a, n', naive_tests)
    def test_zipfian_naive(self, a, n):

        @np.vectorize
        def Hns(n, s):
            """Naive implementation of harmonic sum"""
            return (1 / np.arange(1, n + 1) ** s).sum()

        @np.vectorize
        def pzip(k, a, n):
            """Naive implementation of zipfian pmf"""
            if k < 1 or k > n:
                return 0.0
            else:
                return 1 / k ** a / Hns(n, a)
        k = np.arange(n + 1)
        pmf = pzip(k, a, n)
        cdf = np.cumsum(pmf)
        mean = np.average(k, weights=pmf)
        var = np.average((k - mean) ** 2, weights=pmf)
        std = var ** 0.5
        skew = np.average(((k - mean) / std) ** 3, weights=pmf)
        kurtosis = np.average(((k - mean) / std) ** 4, weights=pmf) - 3
        assert_allclose(zipfian.pmf(k, a, n), pmf)
        assert_allclose(zipfian.cdf(k, a, n), cdf)
        assert_allclose(zipfian.stats(a, n, moments='mvsk'), [mean, var, skew, kurtosis])