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
class TestSpecialOrthoGroup:

    def test_reproducibility(self):
        np.random.seed(514)
        x = special_ortho_group.rvs(3)
        expected = np.array([[-0.99394515, -0.04527879, 0.10011432], [0.04821555, -0.99846897, 0.02711042], [0.09873351, 0.03177334, 0.99460653]])
        assert_array_almost_equal(x, expected)
        random_state = np.random.RandomState(seed=514)
        x = special_ortho_group.rvs(3, random_state=random_state)
        assert_array_almost_equal(x, expected)

    def test_invalid_dim(self):
        assert_raises(ValueError, special_ortho_group.rvs, None)
        assert_raises(ValueError, special_ortho_group.rvs, (2, 2))
        assert_raises(ValueError, special_ortho_group.rvs, 1)
        assert_raises(ValueError, special_ortho_group.rvs, 2.5)

    def test_frozen_matrix(self):
        dim = 7
        frozen = special_ortho_group(dim)
        rvs1 = frozen.rvs(random_state=1234)
        rvs2 = special_ortho_group.rvs(dim, random_state=1234)
        assert_equal(rvs1, rvs2)

    def test_det_and_ortho(self):
        xs = [special_ortho_group.rvs(dim) for dim in range(2, 12) for i in range(3)]
        dets = [np.linalg.det(x) for x in xs]
        assert_allclose(dets, [1.0] * 30, rtol=1e-13)
        for x in xs:
            assert_array_almost_equal(np.dot(x, x.T), np.eye(x.shape[0]))

    def test_haar(self):
        dim = 5
        samples = 1000
        ks_prob = 0.05
        np.random.seed(514)
        xs = special_ortho_group.rvs(dim, size=samples)
        els = ((0, 0), (0, 2), (1, 4), (2, 3))
        proj = {(er, ec): sorted([x[er][ec] for x in xs]) for er, ec in els}
        pairs = [(e0, e1) for e0 in els for e1 in els if e0 > e1]
        ks_tests = [ks_2samp(proj[p0], proj[p1])[1] for p0, p1 in pairs]
        assert_array_less([ks_prob] * len(pairs), ks_tests)