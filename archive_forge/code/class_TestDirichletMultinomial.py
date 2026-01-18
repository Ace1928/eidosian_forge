import pickle
from numpy.testing import (assert_allclose, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
from .test_continuous_basic import check_distribution_rvs
import numpy
import numpy as np
import scipy.linalg
from scipy.stats._multivariate import (_PSD,
from scipy.stats import (multivariate_normal, multivariate_hypergeom,
from scipy.stats import _covariance, Covariance
from scipy import stats
from scipy.integrate import romb, qmc_quad, tplquad
from scipy.special import multigammaln
from scipy._lib._pep440 import Version
from .common_tests import check_random_state_property
from .data._mvt import _qsimvtv
from unittest.mock import patch
class TestDirichletMultinomial:

    @classmethod
    def get_params(self, m):
        rng = np.random.default_rng(28469824356873456)
        alpha = rng.uniform(0, 100, size=2)
        x = rng.integers(1, 20, size=(m, 2))
        n = x.sum(axis=-1)
        return (rng, m, alpha, n, x)

    def test_frozen(self):
        rng = np.random.default_rng(28469824356873456)
        alpha = rng.uniform(0, 100, 10)
        x = rng.integers(0, 10, 10)
        n = np.sum(x, axis=-1)
        d = dirichlet_multinomial(alpha, n)
        assert_equal(d.logpmf(x), dirichlet_multinomial.logpmf(x, alpha, n))
        assert_equal(d.pmf(x), dirichlet_multinomial.pmf(x, alpha, n))
        assert_equal(d.mean(), dirichlet_multinomial.mean(alpha, n))
        assert_equal(d.var(), dirichlet_multinomial.var(alpha, n))
        assert_equal(d.cov(), dirichlet_multinomial.cov(alpha, n))

    def test_pmf_logpmf_against_R(self):
        x = np.array([1, 2, 3])
        n = np.sum(x)
        alpha = np.array([3, 4, 5])
        res = dirichlet_multinomial.pmf(x, alpha, n)
        logres = dirichlet_multinomial.logpmf(x, alpha, n)
        ref = 0.08484162895927638
        assert_allclose(res, ref)
        assert_allclose(logres, np.log(ref))
        assert res.shape == logres.shape == ()
        rng = np.random.default_rng(28469824356873456)
        alpha = rng.uniform(0, 100, 10)
        x = rng.integers(0, 10, 10)
        n = np.sum(x, axis=-1)
        res = dirichlet_multinomial(alpha, n).pmf(x)
        logres = dirichlet_multinomial.logpmf(x, alpha, n)
        ref = 3.65409306285992e-16
        assert_allclose(res, ref)
        assert_allclose(logres, np.log(ref))

    def test_pmf_logpmf_support(self):
        rng, m, alpha, n, x = self.get_params(1)
        n += 1
        assert_equal(dirichlet_multinomial(alpha, n).pmf(x), 0)
        assert_equal(dirichlet_multinomial(alpha, n).logpmf(x), -np.inf)
        rng, m, alpha, n, x = self.get_params(10)
        i = rng.random(size=10) > 0.5
        x[i] = np.round(x[i] * 2)
        assert_equal(dirichlet_multinomial(alpha, n).pmf(x)[i], 0)
        assert_equal(dirichlet_multinomial(alpha, n).logpmf(x)[i], -np.inf)
        assert np.all(dirichlet_multinomial(alpha, n).pmf(x)[~i] > 0)
        assert np.all(dirichlet_multinomial(alpha, n).logpmf(x)[~i] > -np.inf)

    def test_dimensionality_one(self):
        n = 6
        alpha = [10]
        x = np.asarray([n])
        dist = dirichlet_multinomial(alpha, n)
        assert_equal(dist.pmf(x), 1)
        assert_equal(dist.pmf(x + 1), 0)
        assert_equal(dist.logpmf(x), 0)
        assert_equal(dist.logpmf(x + 1), -np.inf)
        assert_equal(dist.mean(), n)
        assert_equal(dist.var(), 0)
        assert_equal(dist.cov(), 0)

    @pytest.mark.parametrize('method_name', ['pmf', 'logpmf'])
    def test_against_betabinom_pmf(self, method_name):
        rng, m, alpha, n, x = self.get_params(100)
        method = getattr(dirichlet_multinomial(alpha, n), method_name)
        ref_method = getattr(stats.betabinom(n, *alpha.T), method_name)
        res = method(x)
        ref = ref_method(x.T[0])
        assert_allclose(res, ref)

    @pytest.mark.parametrize('method_name', ['mean', 'var'])
    def test_against_betabinom_moments(self, method_name):
        rng, m, alpha, n, x = self.get_params(100)
        method = getattr(dirichlet_multinomial(alpha, n), method_name)
        ref_method = getattr(stats.betabinom(n, *alpha.T), method_name)
        res = method()[:, 0]
        ref = ref_method()
        assert_allclose(res, ref)

    def test_moments(self):
        message = 'Needs NumPy 1.22.0 for multinomial broadcasting'
        if Version(np.__version__) < Version('1.22.0'):
            pytest.skip(reason=message)
        rng = np.random.default_rng(28469824356873456)
        dim = 5
        n = rng.integers(1, 100)
        alpha = rng.random(size=dim) * 10
        dist = dirichlet_multinomial(alpha, n)
        m = 100000
        p = rng.dirichlet(alpha, size=m)
        x = rng.multinomial(n, p, size=m)
        assert_allclose(dist.mean(), np.mean(x, axis=0), rtol=0.005)
        assert_allclose(dist.var(), np.var(x, axis=0), rtol=0.01)
        assert dist.mean().shape == dist.var().shape == (dim,)
        cov = dist.cov()
        assert cov.shape == (dim, dim)
        assert_allclose(cov, np.cov(x.T), rtol=0.02)
        assert_equal(np.diag(cov), dist.var())
        assert np.all(scipy.linalg.eigh(cov)[0] > 0)

    def test_input_validation(self):
        x0 = np.array([1, 2, 3])
        n0 = np.sum(x0)
        alpha0 = np.array([3, 4, 5])
        text = '`x` must contain only non-negative integers.'
        with assert_raises(ValueError, match=text):
            dirichlet_multinomial.logpmf([1, -1, 3], alpha0, n0)
        with assert_raises(ValueError, match=text):
            dirichlet_multinomial.logpmf([1, 2.1, 3], alpha0, n0)
        text = '`alpha` must contain only positive values.'
        with assert_raises(ValueError, match=text):
            dirichlet_multinomial.logpmf(x0, [3, 0, 4], n0)
        with assert_raises(ValueError, match=text):
            dirichlet_multinomial.logpmf(x0, [3, -1, 4], n0)
        text = '`n` must be a positive integer.'
        with assert_raises(ValueError, match=text):
            dirichlet_multinomial.logpmf(x0, alpha0, 49.1)
        with assert_raises(ValueError, match=text):
            dirichlet_multinomial.logpmf(x0, alpha0, 0)
        x = np.array([1, 2, 3, 4])
        alpha = np.array([3, 4, 5])
        text = '`x` and `alpha` must be broadcastable.'
        with assert_raises(ValueError, match=text):
            dirichlet_multinomial.logpmf(x, alpha, x.sum())

    @pytest.mark.parametrize('method', ['pmf', 'logpmf'])
    def test_broadcasting_pmf(self, method):
        alpha = np.array([[3, 4, 5], [4, 5, 6], [5, 5, 7], [8, 9, 10]])
        n = np.array([[6], [7], [8]])
        x = np.array([[1, 2, 3], [2, 2, 3]]).reshape((2, 1, 1, 3))
        method = getattr(dirichlet_multinomial, method)
        res = method(x, alpha, n)
        assert res.shape == (2, 3, 4)
        for i in range(len(x)):
            for j in range(len(n)):
                for k in range(len(alpha)):
                    res_ijk = res[i, j, k]
                    ref = method(x[i].squeeze(), alpha[k].squeeze(), n[j].squeeze())
                    assert_allclose(res_ijk, ref)

    @pytest.mark.parametrize('method_name', ['mean', 'var', 'cov'])
    def test_broadcasting_moments(self, method_name):
        alpha = np.array([[3, 4, 5], [4, 5, 6], [5, 5, 7], [8, 9, 10]])
        n = np.array([[6], [7], [8]])
        method = getattr(dirichlet_multinomial, method_name)
        res = method(alpha, n)
        assert res.shape == (3, 4, 3) if method_name != 'cov' else (3, 4, 3, 3)
        for j in range(len(n)):
            for k in range(len(alpha)):
                res_ijk = res[j, k]
                ref = method(alpha[k].squeeze(), n[j].squeeze())
                assert_allclose(res_ijk, ref)