import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
class TestNCH:
    np.random.seed(2)
    shape = (2, 4, 3)
    max_m = 100
    m1 = np.random.randint(1, max_m, size=shape)
    m2 = np.random.randint(1, max_m, size=shape)
    N = m1 + m2
    n = randint.rvs(0, N, size=N.shape)
    xl = np.maximum(0, n - m2)
    xu = np.minimum(n, m1)
    x = randint.rvs(xl, xu, size=xl.shape)
    odds = np.random.rand(*x.shape) * 2

    @pytest.mark.parametrize('dist_name', ['nchypergeom_fisher', 'nchypergeom_wallenius'])
    def test_nch_hypergeom(self, dist_name):
        dists = {'nchypergeom_fisher': nchypergeom_fisher, 'nchypergeom_wallenius': nchypergeom_wallenius}
        dist = dists[dist_name]
        x, N, m1, n = (self.x, self.N, self.m1, self.n)
        assert_allclose(dist.pmf(x, N, m1, n, odds=1), hypergeom.pmf(x, N, m1, n))

    def test_nchypergeom_fisher_naive(self):
        x, N, m1, n, odds = (self.x, self.N, self.m1, self.n, self.odds)

        @np.vectorize
        def pmf_mean_var(x, N, m1, n, w):
            m2 = N - m1
            xl = np.maximum(0, n - m2)
            xu = np.minimum(n, m1)

            def f(x):
                t1 = special_binom(m1, x)
                t2 = special_binom(m2, n - x)
                return t1 * t2 * w ** x

            def P(k):
                return sum((f(y) * y ** k for y in range(xl, xu + 1)))
            P0 = P(0)
            P1 = P(1)
            P2 = P(2)
            pmf = f(x) / P0
            mean = P1 / P0
            var = P2 / P0 - (P1 / P0) ** 2
            return (pmf, mean, var)
        pmf, mean, var = pmf_mean_var(x, N, m1, n, odds)
        assert_allclose(nchypergeom_fisher.pmf(x, N, m1, n, odds), pmf)
        assert_allclose(nchypergeom_fisher.stats(N, m1, n, odds, moments='m'), mean)
        assert_allclose(nchypergeom_fisher.stats(N, m1, n, odds, moments='v'), var)

    def test_nchypergeom_wallenius_naive(self):
        np.random.seed(2)
        shape = (2, 4, 3)
        max_m = 100
        m1 = np.random.randint(1, max_m, size=shape)
        m2 = np.random.randint(1, max_m, size=shape)
        N = m1 + m2
        n = randint.rvs(0, N, size=N.shape)
        xl = np.maximum(0, n - m2)
        xu = np.minimum(n, m1)
        x = randint.rvs(xl, xu, size=xl.shape)
        w = np.random.rand(*x.shape) * 2

        def support(N, m1, n, w):
            m2 = N - m1
            xl = np.maximum(0, n - m2)
            xu = np.minimum(n, m1)
            return (xl, xu)

        @np.vectorize
        def mean(N, m1, n, w):
            m2 = N - m1
            xl, xu = support(N, m1, n, w)

            def fun(u):
                return u / m1 + (1 - (n - u) / m2) ** w - 1
            return root_scalar(fun, bracket=(xl, xu)).root
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, message='invalid value encountered in mean')
            assert_allclose(nchypergeom_wallenius.mean(N, m1, n, w), mean(N, m1, n, w), rtol=0.02)

        @np.vectorize
        def variance(N, m1, n, w):
            m2 = N - m1
            u = mean(N, m1, n, w)
            a = u * (m1 - u)
            b = (n - u) * (u + m2 - n)
            return N * a * b / ((N - 1) * (m1 * b + m2 * a))
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, message='invalid value encountered in mean')
            assert_allclose(nchypergeom_wallenius.stats(N, m1, n, w, moments='v'), variance(N, m1, n, w), rtol=0.05)

        @np.vectorize
        def pmf(x, N, m1, n, w):
            m2 = N - m1
            xl, xu = support(N, m1, n, w)

            def integrand(t):
                D = w * (m1 - x) + (m2 - (n - x))
                res = (1 - t ** (w / D)) ** x * (1 - t ** (1 / D)) ** (n - x)
                return res

            def f(x):
                t1 = special_binom(m1, x)
                t2 = special_binom(m2, n - x)
                the_integral = quad(integrand, 0, 1, epsrel=1e-16, epsabs=1e-16)
                return t1 * t2 * the_integral[0]
            return f(x)
        pmf0 = pmf(x, N, m1, n, w)
        pmf1 = nchypergeom_wallenius.pmf(x, N, m1, n, w)
        atol, rtol = (1e-06, 1e-06)
        i = np.abs(pmf1 - pmf0) < atol + rtol * np.abs(pmf0)
        assert i.sum() > np.prod(shape) / 2
        for N, m1, n, w in zip(N[~i], m1[~i], n[~i], w[~i]):
            m2 = N - m1
            xl, xu = support(N, m1, n, w)
            x = np.arange(xl, xu + 1)
            assert pmf(x, N, m1, n, w).sum() < 0.5
            assert_allclose(nchypergeom_wallenius.pmf(x, N, m1, n, w).sum(), 1)

    def test_wallenius_against_mpmath(self):
        M = 50
        n = 30
        N = 20
        odds = 2.25
        sup = np.arange(21)
        pmf = np.array([3.699003068656875e-20, 5.89398584245431e-17, 2.1594437742911123e-14, 3.221458044649955e-12, 2.4658279241205077e-10, 1.0965862603981212e-08, 3.057890479665704e-07, 5.622818831643761e-06, 7.056482841531681e-05, 0.000618899425358671, 0.003854172932571669, 0.01720592676256026, 0.05528844897093792, 0.12772363313574242, 0.21065898367825722, 0.24465958845359234, 0.1955114898110033, 0.10355390084949237, 0.03414490375225675, 0.006231989845775931, 0.0004715577304677075])
        mean = 14.808018384813426
        var = 2.6085975877923717
        assert_allclose(nchypergeom_wallenius.pmf(sup, M, n, N, odds), pmf, rtol=1e-13, atol=1e-13)
        assert_allclose(nchypergeom_wallenius.mean(M, n, N, odds), mean, rtol=1e-13)
        assert_allclose(nchypergeom_wallenius.var(M, n, N, odds), var, rtol=1e-11)

    @pytest.mark.parametrize('dist_name', ['nchypergeom_fisher', 'nchypergeom_wallenius'])
    def test_rvs_shape(self, dist_name):
        dists = {'nchypergeom_fisher': nchypergeom_fisher, 'nchypergeom_wallenius': nchypergeom_wallenius}
        dist = dists[dist_name]
        x = dist.rvs(50, 30, [[10], [20]], [0.5, 1.0, 2.0], size=(5, 1, 2, 3))
        assert x.shape == (5, 1, 2, 3)