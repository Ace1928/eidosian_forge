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
class TestUnitaryGroup:

    def test_reproducibility(self):
        np.random.seed(514)
        x = unitary_group.rvs(3)
        x2 = unitary_group.rvs(3, random_state=514)
        expected = np.array([[0.308771 + 0.360312j, 0.044021 + 0.622082j, 0.160327 + 0.600173j], [0.732757 + 0.297107j, 0.076692 - 0.4614j, -0.394349 + 0.022613j], [-0.148844 + 0.357037j, -0.284602 - 0.557949j, 0.607051 + 0.299257j]])
        assert_array_almost_equal(x, expected)
        assert_array_almost_equal(x2, expected)

    def test_invalid_dim(self):
        assert_raises(ValueError, unitary_group.rvs, None)
        assert_raises(ValueError, unitary_group.rvs, (2, 2))
        assert_raises(ValueError, unitary_group.rvs, 1)
        assert_raises(ValueError, unitary_group.rvs, 2.5)

    def test_frozen_matrix(self):
        dim = 7
        frozen = unitary_group(dim)
        frozen_seed = unitary_group(dim, seed=514)
        rvs1 = frozen.rvs(random_state=514)
        rvs2 = unitary_group.rvs(dim, random_state=514)
        rvs3 = frozen_seed.rvs(size=1)
        assert_equal(rvs1, rvs2)
        assert_equal(rvs1, rvs3)

    def test_unitarity(self):
        xs = [unitary_group.rvs(dim) for dim in range(2, 12) for i in range(3)]
        for x in xs:
            assert_allclose(np.dot(x, x.conj().T), np.eye(x.shape[0]), atol=1e-15)

    def test_haar(self):
        dim = 5
        samples = 1000
        np.random.seed(514)
        xs = unitary_group.rvs(dim, size=samples)
        eigs = np.vstack([scipy.linalg.eigvals(x) for x in xs])
        x = np.arctan2(eigs.imag, eigs.real)
        res = kstest(x.ravel(), uniform(-np.pi, 2 * np.pi).cdf)
        assert_(res.pvalue > 0.05)