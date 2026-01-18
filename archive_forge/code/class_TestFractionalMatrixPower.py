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
class TestFractionalMatrixPower:

    def test_round_trip_random_complex(self):
        np.random.seed(1234)
        for p in range(1, 5):
            for n in range(1, 5):
                M_unscaled = np.random.randn(n, n) + 1j * np.random.randn(n, n)
                for scale in np.logspace(-4, 4, 9):
                    M = M_unscaled * scale
                    M_root = fractional_matrix_power(M, 1 / p)
                    M_round_trip = np.linalg.matrix_power(M_root, p)
                    assert_allclose(M_round_trip, M)

    def test_round_trip_random_float(self):
        np.random.seed(1234)
        for p in range(1, 5):
            for n in range(1, 5):
                M_unscaled = np.random.randn(n, n)
                for scale in np.logspace(-4, 4, 9):
                    M = M_unscaled * scale
                    M_root = fractional_matrix_power(M, 1 / p)
                    M_round_trip = np.linalg.matrix_power(M_root, p)
                    assert_allclose(M_round_trip, M)

    def test_larger_abs_fractional_matrix_powers(self):
        np.random.seed(1234)
        for n in (2, 3, 5):
            for i in range(10):
                M = np.random.randn(n, n) + 1j * np.random.randn(n, n)
                M_one_fifth = fractional_matrix_power(M, 0.2)
                M_round_trip = np.linalg.matrix_power(M_one_fifth, 5)
                assert_allclose(M, M_round_trip)
                X = fractional_matrix_power(M, -5.4)
                Y = np.linalg.matrix_power(M_one_fifth, -27)
                assert_allclose(X, Y)
                X = fractional_matrix_power(M, 3.8)
                Y = np.linalg.matrix_power(M_one_fifth, 19)
                assert_allclose(X, Y)

    def test_random_matrices_and_powers(self):
        np.random.seed(1234)
        nsamples = 20
        for i in range(nsamples):
            n = random.randrange(1, 5)
            p = np.random.randn()
            matrix_scale = np.exp(random.randrange(-4, 5))
            A = np.random.randn(n, n)
            if random.choice((True, False)):
                A = A + 1j * np.random.randn(n, n)
            A = A * matrix_scale
            A_power = fractional_matrix_power(A, p)
            A_logm, info = logm(A, disp=False)
            A_power_expm_logm = expm(A_logm * p)
            assert_allclose(A_power, A_power_expm_logm)

    def test_al_mohy_higham_2012_experiment_1(self):
        A = _get_al_mohy_higham_2012_experiment_1()
        A_funm_sqrt, info = funm(A, np.sqrt, disp=False)
        A_sqrtm, info = sqrtm(A, disp=False)
        A_rem_power = _matfuncs_inv_ssq._remainder_matrix_power(A, 0.5)
        A_power = fractional_matrix_power(A, 0.5)
        assert_allclose(A_rem_power, A_power, rtol=1e-11)
        assert_allclose(A_sqrtm, A_power)
        assert_allclose(A_sqrtm, A_funm_sqrt)
        for p in (1 / 2, 5 / 3):
            A_power = fractional_matrix_power(A, p)
            A_round_trip = fractional_matrix_power(A_power, 1 / p)
            assert_allclose(A_round_trip, A, rtol=0.01)
            assert_allclose(np.tril(A_round_trip, 1), np.tril(A, 1))

    def test_briggs_helper_function(self):
        np.random.seed(1234)
        for a in np.random.randn(10) + 1j * np.random.randn(10):
            for k in range(5):
                x_observed = _matfuncs_inv_ssq._briggs_helper_function(a, k)
                x_expected = a ** np.exp2(-k) - 1
                assert_allclose(x_observed, x_expected)

    def test_type_preservation_and_conversion(self):
        complex_dtype_chars = ('F', 'D', 'G')
        for matrix_as_list in ([[1, 0], [0, 1]], [[1, 0], [1, 1]], [[2, 1], [1, 1]], [[2, 3], [1, 2]]):
            W = scipy.linalg.eigvals(matrix_as_list)
            assert_(not any((w.imag or w.real < 0 for w in W)))
            for p in (-2.4, -0.9, 0.2, 3.3):
                A = np.array(matrix_as_list, dtype=float)
                A_power = fractional_matrix_power(A, p)
                assert_(A_power.dtype.char not in complex_dtype_chars)
                A = np.array(matrix_as_list, dtype=complex)
                A_power = fractional_matrix_power(A, p)
                assert_(A_power.dtype.char in complex_dtype_chars)
                A = -np.array(matrix_as_list, dtype=float)
                A_power = fractional_matrix_power(A, p)
                assert_(A_power.dtype.char in complex_dtype_chars)

    def test_type_conversion_mixed_sign_or_complex_spectrum(self):
        complex_dtype_chars = ('F', 'D', 'G')
        for matrix_as_list in ([[1, 0], [0, -1]], [[0, 1], [1, 0]], [[0, 1, 0], [0, 0, 1], [1, 0, 0]]):
            W = scipy.linalg.eigvals(matrix_as_list)
            assert_(any((w.imag or w.real < 0 for w in W)))
            for p in (-2.4, -0.9, 0.2, 3.3):
                A = np.array(matrix_as_list, dtype=complex)
                A_power = fractional_matrix_power(A, p)
                assert_(A_power.dtype.char in complex_dtype_chars)
                A = np.array(matrix_as_list, dtype=float)
                A_power = fractional_matrix_power(A, p)
                assert_(A_power.dtype.char in complex_dtype_chars)

    @pytest.mark.xfail(reason='Too unstable across LAPACKs.')
    def test_singular(self):
        for matrix_as_list in ([[0, 0], [0, 0]], [[1, 1], [1, 1]], [[1, 2], [3, 6]], [[0, 0, 0], [0, 1, 1], [0, -1, 1]]):
            for newtype in (float, complex):
                A = np.array(matrix_as_list, dtype=newtype)
                for p in (-0.7, -0.9, -2.4, -1.3):
                    A_power = fractional_matrix_power(A, p)
                    assert_(np.isnan(A_power).all())
                for p in (0.2, 1.43):
                    A_power = fractional_matrix_power(A, p)
                    A_round_trip = fractional_matrix_power(A_power, 1 / p)
                    assert_allclose(A_round_trip, A)

    def test_opposite_sign_complex_eigenvalues(self):
        M = [[2j, 4], [0, -2j]]
        R = [[1 + 1j, 2], [0, 1 - 1j]]
        assert_allclose(np.dot(R, R), M, atol=1e-14)
        assert_allclose(fractional_matrix_power(M, 0.5), R, atol=1e-14)