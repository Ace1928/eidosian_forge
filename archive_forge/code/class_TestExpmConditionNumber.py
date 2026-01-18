import random
import functools
import numpy as np
from numpy import array, identity, dot, sqrt
from numpy.testing import (assert_array_almost_equal, assert_allclose, assert_,
import pytest
import scipy.linalg
from scipy.linalg import (funm, signm, logm, sqrtm, fractional_matrix_power,
from scipy.linalg import _matfuncs_inv_ssq
import scipy.linalg._expm_frechet
from scipy.optimize import minimize
class TestExpmConditionNumber:

    def test_expm_cond_smoke(self):
        np.random.seed(1234)
        for n in range(1, 4):
            A = np.random.randn(n, n)
            kappa = expm_cond(A)
            assert_array_less(0, kappa)

    def test_expm_bad_condition_number(self):
        A = np.array([[-1.12867982, 96141.83771, -4524855739.0, 292496941100000.0], [0, -1.201010529, 96346.96872, -4681048289.0], [0, 0, -1.132893222, 95324.9183], [0, 0, 0, -1.179475332]])
        kappa = expm_cond(A)
        assert_array_less(1e+36, kappa)

    def test_univariate(self):
        np.random.seed(12345)
        for x in np.linspace(-5, 5, num=11):
            A = np.array([[x]])
            assert_allclose(expm_cond(A), abs(x))
        for x in np.logspace(-2, 2, num=11):
            A = np.array([[x]])
            assert_allclose(expm_cond(A), abs(x))
        for i in range(10):
            A = np.random.randn(1, 1)
            assert_allclose(expm_cond(A), np.absolute(A)[0, 0])

    @pytest.mark.slow
    def test_expm_cond_fuzz(self):
        np.random.seed(12345)
        eps = 1e-05
        nsamples = 10
        for i in range(nsamples):
            n = np.random.randint(2, 5)
            A = np.random.randn(n, n)
            A_norm = scipy.linalg.norm(A)
            X = expm(A)
            X_norm = scipy.linalg.norm(X)
            kappa = expm_cond(A)
            f = functools.partial(_help_expm_cond_search, A, A_norm, X, X_norm, eps)
            guess = np.ones(n * n)
            out = minimize(f, guess, method='L-BFGS-B')
            xopt = out.x
            yopt = f(xopt)
            p_best = eps * _normalized_like(np.reshape(xopt, A.shape), A)
            p_best_relerr = _relative_error(expm, A, p_best)
            assert_allclose(p_best_relerr, -yopt * eps)
            for j in range(5):
                p_rand = eps * _normalized_like(np.random.randn(*A.shape), A)
                assert_allclose(norm(p_best), norm(p_rand))
                p_rand_relerr = _relative_error(expm, A, p_rand)
                assert_array_less(p_rand_relerr, p_best_relerr)
            assert_array_less(p_best_relerr, (1 + 2 * eps) * eps * kappa)