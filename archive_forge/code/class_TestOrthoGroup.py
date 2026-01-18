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
class TestOrthoGroup:

    def test_reproducibility(self):
        seed = 514
        np.random.seed(seed)
        x = ortho_group.rvs(3)
        x2 = ortho_group.rvs(3, random_state=seed)
        assert_almost_equal(np.linalg.det(x), -1)
        expected = np.array([[0.381686, -0.090374, 0.919863], [0.905794, -0.161537, -0.391718], [-0.183993, -0.98272, -0.020204]])
        assert_array_almost_equal(x, expected)
        assert_array_almost_equal(x2, expected)

    def test_invalid_dim(self):
        assert_raises(ValueError, ortho_group.rvs, None)
        assert_raises(ValueError, ortho_group.rvs, (2, 2))
        assert_raises(ValueError, ortho_group.rvs, 1)
        assert_raises(ValueError, ortho_group.rvs, 2.5)

    def test_frozen_matrix(self):
        dim = 7
        frozen = ortho_group(dim)
        frozen_seed = ortho_group(dim, seed=1234)
        rvs1 = frozen.rvs(random_state=1234)
        rvs2 = ortho_group.rvs(dim, random_state=1234)
        rvs3 = frozen_seed.rvs(size=1)
        assert_equal(rvs1, rvs2)
        assert_equal(rvs1, rvs3)

    def test_det_and_ortho(self):
        xs = [[ortho_group.rvs(dim) for i in range(10)] for dim in range(2, 12)]
        dets = np.array([[np.linalg.det(x) for x in xx] for xx in xs])
        assert_allclose(np.fabs(dets), np.ones(dets.shape), rtol=1e-13)
        for xx in xs:
            for x in xx:
                assert_array_almost_equal(np.dot(x, x.T), np.eye(x.shape[0]))

    @pytest.mark.parametrize('dim', [2, 5, 10, 20])
    def test_det_distribution_gh18272(self, dim):
        rng = np.random.default_rng(6796248956179332344)
        dist = ortho_group(dim=dim)
        rvs = dist.rvs(size=5000, random_state=rng)
        dets = scipy.linalg.det(rvs)
        k = np.sum(dets > 0)
        n = len(dets)
        res = stats.binomtest(k, n)
        low, high = res.proportion_ci(confidence_level=0.95)
        assert low < 0.5 < high

    def test_haar(self):
        dim = 5
        samples = 1000
        ks_prob = 0.05
        np.random.seed(518)
        xs = ortho_group.rvs(dim, size=samples)
        els = ((0, 0), (0, 2), (1, 4), (2, 3))
        proj = {(er, ec): sorted([x[er][ec] for x in xs]) for er, ec in els}
        pairs = [(e0, e1) for e0 in els for e1 in els if e0 > e1]
        ks_tests = [ks_2samp(proj[p0], proj[p1])[1] for p0, p1 in pairs]
        assert_array_less([ks_prob] * len(pairs), ks_tests)

    @pytest.mark.slow
    def test_pairwise_distances(self):
        np.random.seed(514)

        def random_ortho(dim):
            u, _s, v = np.linalg.svd(np.random.normal(size=(dim, dim)))
            return np.dot(u, v)
        for dim in range(2, 6):

            def generate_test_statistics(rvs, N=1000, eps=1e-10):
                stats = np.array([np.sum((rvs(dim=dim) - rvs(dim=dim)) ** 2) for _ in range(N)])
                stats += np.random.uniform(-eps, eps, size=stats.shape)
                return stats
            expected = generate_test_statistics(random_ortho)
            actual = generate_test_statistics(scipy.stats.ortho_group.rvs)
            _D, p = scipy.stats.ks_2samp(expected, actual)
            assert_array_less(0.05, p)