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
class TestVonMises_Fisher:

    @pytest.mark.parametrize('dim', [2, 3, 4, 6])
    @pytest.mark.parametrize('size', [None, 1, 5, (5, 4)])
    def test_samples(self, dim, size):
        rng = np.random.default_rng(2777937887058094419)
        mu = np.full((dim,), 1 / np.sqrt(dim))
        vmf_dist = vonmises_fisher(mu, 1, seed=rng)
        samples = vmf_dist.rvs(size)
        mean, cov = (np.zeros(dim), np.eye(dim))
        expected_shape = rng.multivariate_normal(mean, cov, size=size).shape
        assert samples.shape == expected_shape
        norms = np.linalg.norm(samples, axis=-1)
        assert_allclose(norms, 1.0)

    @pytest.mark.parametrize('dim', [5, 8])
    @pytest.mark.parametrize('kappa', [1000000000000000.0, 1e+20, 1e+30])
    def test_sampling_high_concentration(self, dim, kappa):
        rng = np.random.default_rng(2777937887058094419)
        mu = np.full((dim,), 1 / np.sqrt(dim))
        vmf_dist = vonmises_fisher(mu, kappa, seed=rng)
        vmf_dist.rvs(10)

    def test_two_dimensional_mu(self):
        mu = np.ones((2, 2))
        msg = "'mu' must have one-dimensional shape."
        with pytest.raises(ValueError, match=msg):
            vonmises_fisher(mu, 1)

    def test_wrong_norm_mu(self):
        mu = np.ones((2,))
        msg = "'mu' must be a unit vector of norm 1."
        with pytest.raises(ValueError, match=msg):
            vonmises_fisher(mu, 1)

    def test_one_entry_mu(self):
        mu = np.ones((1,))
        msg = "'mu' must have at least two entries."
        with pytest.raises(ValueError, match=msg):
            vonmises_fisher(mu, 1)

    @pytest.mark.parametrize('kappa', [-1, (5, 3)])
    def test_kappa_validation(self, kappa):
        msg = "'kappa' must be a positive scalar."
        with pytest.raises(ValueError, match=msg):
            vonmises_fisher([1, 0], kappa)

    @pytest.mark.parametrize('kappa', [0, 0.0])
    def test_kappa_zero(self, kappa):
        msg = "For 'kappa=0' the von Mises-Fisher distribution becomes the uniform distribution on the sphere surface. Consider using 'scipy.stats.uniform_direction' instead."
        with pytest.raises(ValueError, match=msg):
            vonmises_fisher([1, 0], kappa)

    @pytest.mark.parametrize('method', [vonmises_fisher.pdf, vonmises_fisher.logpdf])
    def test_invalid_shapes_pdf_logpdf(self, method):
        x = np.array([1.0, 0.0, 0])
        msg = "The dimensionality of the last axis of 'x' must match the dimensionality of the von Mises Fisher distribution."
        with pytest.raises(ValueError, match=msg):
            method(x, [1, 0], 1)

    @pytest.mark.parametrize('method', [vonmises_fisher.pdf, vonmises_fisher.logpdf])
    def test_unnormalized_input(self, method):
        x = np.array([0.5, 0.0])
        msg = "'x' must be unit vectors of norm 1 along last dimension."
        with pytest.raises(ValueError, match=msg):
            method(x, [1, 0], 1)

    @pytest.mark.parametrize('x, mu, kappa, reference', [(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.0001, 0.0795854295583605), (np.array([1.0, 0.0, 0]), np.array([0.0, 0.0, 1.0]), 0.0001, 0.07957747141331854), (np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 100, 15.915494309189533), (np.array([1.0, 0.0, 0]), np.array([0.0, 0.0, 1.0]), 100, 5.920684802611232e-43), (np.array([1.0, 0.0, 0.0]), np.array([np.sqrt(0.98), np.sqrt(0.02), 0.0]), 2000, 5.930499050746588e-07), (np.array([1.0, 0.0, 0]), np.array([1.0, 0.0, 0.0]), 2000, 318.3098861837907), (np.array([1.0, 0.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0, 0.0]), 2000, 101371.86957712633), (np.array([1.0, 0.0, 0.0, 0.0, 0.0]), np.array([np.sqrt(0.98), np.sqrt(0.02), 0.0, 0, 0.0]), 2000, 0.00018886808182653578), (np.array([1.0, 0.0, 0.0, 0.0, 0.0]), np.array([np.sqrt(0.8), np.sqrt(0.2), 0.0, 0, 0.0]), 2000, 2.0255393314603194e-87)])
    def test_pdf_accuracy(self, x, mu, kappa, reference):
        pdf = vonmises_fisher(mu, kappa).pdf(x)
        assert_allclose(pdf, reference, rtol=1e-13)

    @pytest.mark.parametrize('x, mu, kappa, reference', [(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.0001, -2.5309242486359573), (np.array([1.0, 0.0, 0]), np.array([0.0, 0.0, 1.0]), 0.0001, -2.5310242486359575), (np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 100, 2.767293119578746), (np.array([1.0, 0.0, 0]), np.array([0.0, 0.0, 1.0]), 100, -97.23270688042125), (np.array([1.0, 0.0, 0.0]), np.array([np.sqrt(0.98), np.sqrt(0.02), 0.0]), 2000, -14.337987284534103), (np.array([1.0, 0.0, 0]), np.array([1.0, 0.0, 0.0]), 2000, 5.763025393132737), (np.array([1.0, 0.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0, 0.0]), 2000, 11.526550911307156), (np.array([1.0, 0.0, 0.0, 0.0, 0.0]), np.array([np.sqrt(0.98), np.sqrt(0.02), 0.0, 0, 0.0]), 2000, -8.574461766359684), (np.array([1.0, 0.0, 0.0, 0.0, 0.0]), np.array([np.sqrt(0.8), np.sqrt(0.2), 0.0, 0, 0.0]), 2000, -199.61906708886113)])
    def test_logpdf_accuracy(self, x, mu, kappa, reference):
        logpdf = vonmises_fisher(mu, kappa).logpdf(x)
        assert_allclose(logpdf, reference, rtol=1e-14)

    @pytest.mark.parametrize('dim, kappa, reference', [(3, 0.0001, 2.531024245302624), (3, 100, -1.7672931195787458), (5, 5000, -11.359032310024453), (8, 1, 3.4189526482545527)])
    def test_entropy_accuracy(self, dim, kappa, reference):
        mu = np.full((dim,), 1 / np.sqrt(dim))
        entropy = vonmises_fisher(mu, kappa).entropy()
        assert_allclose(entropy, reference, rtol=2e-14)

    @pytest.mark.parametrize('method', [vonmises_fisher.pdf, vonmises_fisher.logpdf])
    def test_broadcasting(self, method):
        testshape = (2, 2)
        rng = np.random.default_rng(2777937887058094419)
        x = uniform_direction(3).rvs(testshape, random_state=rng)
        mu = np.full((3,), 1 / np.sqrt(3))
        kappa = 5
        result_all = method(x, mu, kappa)
        assert result_all.shape == testshape
        for i in range(testshape[0]):
            for j in range(testshape[1]):
                current_val = method(x[i, j, :], mu, kappa)
                assert_allclose(current_val, result_all[i, j], rtol=1e-15)

    def test_vs_vonmises_2d(self):
        rng = np.random.default_rng(2777937887058094419)
        mu = np.array([0, 1])
        mu_angle = np.arctan2(mu[1], mu[0])
        kappa = 20
        vmf = vonmises_fisher(mu, kappa)
        vonmises_dist = vonmises(loc=mu_angle, kappa=kappa)
        vectors = uniform_direction(2).rvs(10, random_state=rng)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        assert_allclose(vonmises_dist.entropy(), vmf.entropy())
        assert_allclose(vonmises_dist.pdf(angles), vmf.pdf(vectors))
        assert_allclose(vonmises_dist.logpdf(angles), vmf.logpdf(vectors))

    @pytest.mark.parametrize('dim', [2, 3, 6])
    @pytest.mark.parametrize('kappa, mu_tol, kappa_tol', [(1, 0.05, 0.05), (10, 0.01, 0.01), (100, 0.005, 0.02), (1000, 0.001, 0.02)])
    def test_fit_accuracy(self, dim, kappa, mu_tol, kappa_tol):
        mu = np.full((dim,), 1 / np.sqrt(dim))
        vmf_dist = vonmises_fisher(mu, kappa)
        rng = np.random.default_rng(2777937887058094419)
        n_samples = 10000
        samples = vmf_dist.rvs(n_samples, random_state=rng)
        mu_fit, kappa_fit = vonmises_fisher.fit(samples)
        angular_error = np.arccos(mu.dot(mu_fit))
        assert_allclose(angular_error, 0.0, atol=mu_tol, rtol=0)
        assert_allclose(kappa, kappa_fit, rtol=kappa_tol)

    def test_fit_error_one_dimensional_data(self):
        x = np.zeros((3,))
        msg = "'x' must be two dimensional."
        with pytest.raises(ValueError, match=msg):
            vonmises_fisher.fit(x)

    def test_fit_error_unnormalized_data(self):
        x = np.ones((3, 3))
        msg = "'x' must be unit vectors of norm 1 along last dimension."
        with pytest.raises(ValueError, match=msg):
            vonmises_fisher.fit(x)

    def test_frozen_distribution(self):
        mu = np.array([0, 0, 1])
        kappa = 5
        frozen = vonmises_fisher(mu, kappa)
        frozen_seed = vonmises_fisher(mu, kappa, seed=514)
        rvs1 = frozen.rvs(random_state=514)
        rvs2 = vonmises_fisher.rvs(mu, kappa, random_state=514)
        rvs3 = frozen_seed.rvs()
        assert_equal(rvs1, rvs2)
        assert_equal(rvs1, rvs3)