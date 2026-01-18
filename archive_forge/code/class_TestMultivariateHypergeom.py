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
class TestMultivariateHypergeom:

    @pytest.mark.parametrize('x, m, n, expected', [([3, 4], [5, 10], 7, -1.119814), ([3, 4], [5, 10], 0, -np.inf), ([-3, 4], [5, 10], 7, -np.inf), ([3, 4], [-5, 10], 7, np.nan), ([[1, 2], [3, 4]], [[-4, -6], [-5, -10]], [3, 7], [np.nan, np.nan]), ([-3, 4], [-5, 10], 1, np.nan), ([1, 11], [10, 1], 12, np.nan), ([1, 11], [10, -1], 12, np.nan), ([3, 4], [5, 10], -7, np.nan), ([3, 3], [5, 10], 7, -np.inf)])
    def test_logpmf(self, x, m, n, expected):
        vals = multivariate_hypergeom.logpmf(x, m, n)
        assert_allclose(vals, expected, rtol=1e-06)

    def test_reduces_hypergeom(self):
        val1 = multivariate_hypergeom.pmf(x=[3, 1], m=[10, 5], n=4)
        val2 = hypergeom.pmf(k=3, M=15, n=4, N=10)
        assert_allclose(val1, val2, rtol=1e-08)
        val1 = multivariate_hypergeom.pmf(x=[7, 3], m=[15, 10], n=10)
        val2 = hypergeom.pmf(k=7, M=25, n=10, N=15)
        assert_allclose(val1, val2, rtol=1e-08)

    def test_rvs(self):
        rv = multivariate_hypergeom(m=[3, 5], n=4)
        rvs = rv.rvs(size=1000, random_state=123)
        assert_allclose(rvs.mean(0), rv.mean(), rtol=0.01)

    def test_rvs_broadcasting(self):
        rv = multivariate_hypergeom(m=[[3, 5], [5, 10]], n=[4, 9])
        rvs = rv.rvs(size=(1000, 2), random_state=123)
        assert_allclose(rvs.mean(0), rv.mean(), rtol=0.01)

    @pytest.mark.parametrize('m, n', (([0, 0, 20, 0, 0], 5), ([0, 0, 0, 0, 0], 0), ([0, 0], 0), ([0], 0)))
    def test_rvs_gh16171(self, m, n):
        res = multivariate_hypergeom.rvs(m, n)
        m = np.asarray(m)
        res_ex = m.copy()
        res_ex[m != 0] = n
        assert_equal(res, res_ex)

    @pytest.mark.parametrize('x, m, n, expected', [([5], [5], 5, 1), ([3, 4], [5, 10], 7, 0.3263403), ([[[3, 5], [0, 8]], [[-1, 9], [1, 1]]], [5, 10], [[8, 8], [8, 2]], [[0.3916084, 0.006993007], [0, 0.4761905]]), (np.array([], dtype=int), np.array([], dtype=int), 0, []), ([1, 2], [4, 5], 5, 0), ([3, 3, 0], [5, 6, 7], 6, 0.01077354)])
    def test_pmf(self, x, m, n, expected):
        vals = multivariate_hypergeom.pmf(x, m, n)
        assert_allclose(vals, expected, rtol=1e-07)

    @pytest.mark.parametrize('x, m, n, expected', [([3, 4], [[5, 10], [10, 15]], 7, [0.3263403, 0.3407531]), ([[1], [2]], [[3], [4]], [1, 3], [1.0, 0.0]), ([[[1], [2]]], [[3], [4]], [1, 3], [[1.0, 0.0]]), ([[1], [2]], [[[[3]]]], [1, 3], [[[1.0, 0.0]]])])
    def test_pmf_broadcasting(self, x, m, n, expected):
        vals = multivariate_hypergeom.pmf(x, m, n)
        assert_allclose(vals, expected, rtol=1e-07)

    def test_cov(self):
        cov1 = multivariate_hypergeom.cov(m=[3, 7, 10], n=12)
        cov2 = [[0.64421053, -0.26526316, -0.37894737], [-0.26526316, 1.14947368, -0.88421053], [-0.37894737, -0.88421053, 1.26315789]]
        assert_allclose(cov1, cov2, rtol=1e-08)

    def test_cov_broadcasting(self):
        cov1 = multivariate_hypergeom.cov(m=[[7, 9], [10, 15]], n=[8, 12])
        cov2 = [[[1.05, -1.05], [-1.05, 1.05]], [[1.56, -1.56], [-1.56, 1.56]]]
        assert_allclose(cov1, cov2, rtol=1e-08)
        cov3 = multivariate_hypergeom.cov(m=[[4], [5]], n=[4, 5])
        cov4 = [[[0.0]], [[0.0]]]
        assert_allclose(cov3, cov4, rtol=1e-08)
        cov5 = multivariate_hypergeom.cov(m=[7, 9], n=[8, 12])
        cov6 = [[[1.05, -1.05], [-1.05, 1.05]], [[0.7875, -0.7875], [-0.7875, 0.7875]]]
        assert_allclose(cov5, cov6, rtol=1e-08)

    def test_var(self):
        var0 = multivariate_hypergeom.var(m=[10, 5], n=4)
        var1 = hypergeom.var(M=15, n=4, N=10)
        assert_allclose(var0, var1, rtol=1e-08)

    def test_var_broadcasting(self):
        var0 = multivariate_hypergeom.var(m=[10, 5], n=[4, 8])
        var1 = multivariate_hypergeom.var(m=[10, 5], n=4)
        var2 = multivariate_hypergeom.var(m=[10, 5], n=8)
        assert_allclose(var0[0], var1, rtol=1e-08)
        assert_allclose(var0[1], var2, rtol=1e-08)
        var3 = multivariate_hypergeom.var(m=[[10, 5], [10, 14]], n=[4, 8])
        var4 = [[0.6984127, 0.6984127], [1.352657, 1.352657]]
        assert_allclose(var3, var4, rtol=1e-08)
        var5 = multivariate_hypergeom.var(m=[[5], [10]], n=[5, 10])
        var6 = [[0.0], [0.0]]
        assert_allclose(var5, var6, rtol=1e-08)

    def test_mean(self):
        mean0 = multivariate_hypergeom.mean(m=[10, 5], n=4)
        mean1 = hypergeom.mean(M=15, n=4, N=10)
        assert_allclose(mean0[0], mean1, rtol=1e-08)
        mean2 = multivariate_hypergeom.mean(m=[12, 8], n=10)
        mean3 = [12.0 * 10.0 / 20.0, 8.0 * 10.0 / 20.0]
        assert_allclose(mean2, mean3, rtol=1e-08)

    def test_mean_broadcasting(self):
        mean0 = multivariate_hypergeom.mean(m=[[3, 5], [10, 5]], n=[4, 8])
        mean1 = [[3.0 * 4.0 / 8.0, 5.0 * 4.0 / 8.0], [10.0 * 8.0 / 15.0, 5.0 * 8.0 / 15.0]]
        assert_allclose(mean0, mean1, rtol=1e-08)

    def test_mean_edge_cases(self):
        mean0 = multivariate_hypergeom.mean(m=[0, 0, 0], n=0)
        assert_equal(mean0, [0.0, 0.0, 0.0])
        mean1 = multivariate_hypergeom.mean(m=[1, 0, 0], n=2)
        assert_equal(mean1, [np.nan, np.nan, np.nan])
        mean2 = multivariate_hypergeom.mean(m=[[1, 0, 0], [1, 0, 1]], n=2)
        assert_allclose(mean2, [[np.nan, np.nan, np.nan], [1.0, 0.0, 1.0]], rtol=1e-17)
        mean3 = multivariate_hypergeom.mean(m=np.array([], dtype=int), n=0)
        assert_equal(mean3, [])
        assert_(mean3.shape == (0,))

    def test_var_edge_cases(self):
        var0 = multivariate_hypergeom.var(m=[0, 0, 0], n=0)
        assert_allclose(var0, [0.0, 0.0, 0.0], rtol=1e-16)
        var1 = multivariate_hypergeom.var(m=[1, 0, 0], n=2)
        assert_equal(var1, [np.nan, np.nan, np.nan])
        var2 = multivariate_hypergeom.var(m=[[1, 0, 0], [1, 0, 1]], n=2)
        assert_allclose(var2, [[np.nan, np.nan, np.nan], [0.0, 0.0, 0.0]], rtol=1e-17)
        var3 = multivariate_hypergeom.var(m=np.array([], dtype=int), n=0)
        assert_equal(var3, [])
        assert_(var3.shape == (0,))

    def test_cov_edge_cases(self):
        cov0 = multivariate_hypergeom.cov(m=[1, 0, 0], n=1)
        cov1 = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        assert_allclose(cov0, cov1, rtol=1e-17)
        cov3 = multivariate_hypergeom.cov(m=[0, 0, 0], n=0)
        cov4 = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        assert_equal(cov3, cov4)
        cov5 = multivariate_hypergeom.cov(m=np.array([], dtype=int), n=0)
        cov6 = np.array([], dtype=np.float64).reshape(0, 0)
        assert_allclose(cov5, cov6, rtol=1e-17)
        assert_(cov5.shape == (0, 0))

    def test_frozen(self):
        np.random.seed(1234)
        n = 12
        m = [7, 9, 11, 13]
        x = [[0, 0, 0, 12], [0, 0, 1, 11], [0, 1, 1, 10], [1, 1, 1, 9], [1, 1, 2, 8]]
        x = np.asarray(x, dtype=int)
        mhg_frozen = multivariate_hypergeom(m, n)
        assert_allclose(mhg_frozen.pmf(x), multivariate_hypergeom.pmf(x, m, n))
        assert_allclose(mhg_frozen.logpmf(x), multivariate_hypergeom.logpmf(x, m, n))
        assert_allclose(mhg_frozen.var(), multivariate_hypergeom.var(m, n))
        assert_allclose(mhg_frozen.cov(), multivariate_hypergeom.cov(m, n))

    def test_invalid_params(self):
        assert_raises(ValueError, multivariate_hypergeom.pmf, 5, 10, 5)
        assert_raises(ValueError, multivariate_hypergeom.pmf, 5, [10], 5)
        assert_raises(ValueError, multivariate_hypergeom.pmf, [5, 4], [10], 5)
        assert_raises(TypeError, multivariate_hypergeom.pmf, [5.5, 4.5], [10, 15], 5)
        assert_raises(TypeError, multivariate_hypergeom.pmf, [5, 4], [10.5, 15.5], 5)
        assert_raises(TypeError, multivariate_hypergeom.pmf, [5, 4], [10, 15], 5.5)