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
class TestRandomCorrelation:

    def test_reproducibility(self):
        np.random.seed(514)
        eigs = (0.5, 0.8, 1.2, 1.5)
        x = random_correlation.rvs(eigs)
        x2 = random_correlation.rvs(eigs, random_state=514)
        expected = np.array([[1.0, -0.184851, 0.109017, -0.227494], [-0.184851, 1.0, 0.231236, 0.326669], [0.109017, 0.231236, 1.0, -0.178912], [-0.227494, 0.326669, -0.178912, 1.0]])
        assert_array_almost_equal(x, expected)
        assert_array_almost_equal(x2, expected)

    def test_invalid_eigs(self):
        assert_raises(ValueError, random_correlation.rvs, None)
        assert_raises(ValueError, random_correlation.rvs, 'test')
        assert_raises(ValueError, random_correlation.rvs, 2.5)
        assert_raises(ValueError, random_correlation.rvs, [2.5])
        assert_raises(ValueError, random_correlation.rvs, [[1, 2], [3, 4]])
        assert_raises(ValueError, random_correlation.rvs, [2.5, -0.5])
        assert_raises(ValueError, random_correlation.rvs, [1, 2, 0.1])

    def test_frozen_matrix(self):
        eigs = (0.5, 0.8, 1.2, 1.5)
        frozen = random_correlation(eigs)
        frozen_seed = random_correlation(eigs, seed=514)
        rvs1 = random_correlation.rvs(eigs, random_state=514)
        rvs2 = frozen.rvs(random_state=514)
        rvs3 = frozen_seed.rvs()
        assert_equal(rvs1, rvs2)
        assert_equal(rvs1, rvs3)

    def test_definition(self):

        def norm(i, e):
            return i * e / sum(e)
        np.random.seed(123)
        eigs = [norm(i, np.random.uniform(size=i)) for i in range(2, 6)]
        eigs.append([4, 0, 0, 0])
        ones = [[1.0] * len(e) for e in eigs]
        xs = [random_correlation.rvs(e) for e in eigs]
        dets = [np.fabs(np.linalg.det(x)) for x in xs]
        dets_known = [np.prod(e) for e in eigs]
        assert_allclose(dets, dets_known, rtol=1e-13, atol=1e-13)
        diags = [np.diag(x) for x in xs]
        for a, b in zip(diags, ones):
            assert_allclose(a, b, rtol=1e-13)
        for x in xs:
            assert_allclose(x, x.T, rtol=1e-13)

    def test_to_corr(self):
        m = np.array([[0.1, 0], [0, 1]], dtype=float)
        m = random_correlation._to_corr(m)
        assert_allclose(m, np.array([[1, 0], [0, 0.1]]))
        with np.errstate(over='ignore'):
            g = np.array([[0, 1], [-1, 0]])
            m0 = np.array([[1e+300, 0], [0, np.nextafter(1, 0)]], dtype=float)
            m = random_correlation._to_corr(m0.copy())
            assert_allclose(m, g.T.dot(m0).dot(g))
            m0 = np.array([[0.9, 1e+300], [1e+300, 1.1]], dtype=float)
            m = random_correlation._to_corr(m0.copy())
            assert_allclose(m, g.T.dot(m0).dot(g))
        m0 = np.array([[2, 1], [1, 2]], dtype=float)
        m = random_correlation._to_corr(m0.copy())
        assert_allclose(m[0, 0], 1)
        m0 = np.array([[2 + 1e-07, 1], [1, 2]], dtype=float)
        m = random_correlation._to_corr(m0.copy())
        assert_allclose(m[0, 0], 1)