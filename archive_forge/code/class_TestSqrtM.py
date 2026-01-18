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
class TestSqrtM:

    def test_round_trip_random_float(self):
        np.random.seed(1234)
        for n in range(1, 6):
            M_unscaled = np.random.randn(n, n)
            for scale in np.logspace(-4, 4, 9):
                M = M_unscaled * scale
                M_sqrtm, info = sqrtm(M, disp=False)
                M_sqrtm_round_trip = M_sqrtm.dot(M_sqrtm)
                assert_allclose(M_sqrtm_round_trip, M)

    def test_round_trip_random_complex(self):
        np.random.seed(1234)
        for n in range(1, 6):
            M_unscaled = np.random.randn(n, n) + 1j * np.random.randn(n, n)
            for scale in np.logspace(-4, 4, 9):
                M = M_unscaled * scale
                M_sqrtm, info = sqrtm(M, disp=False)
                M_sqrtm_round_trip = M_sqrtm.dot(M_sqrtm)
                assert_allclose(M_sqrtm_round_trip, M)

    def test_bad(self):
        e = 2 ** (-5)
        se = sqrt(e)
        a = array([[1.0, 0, 0, 1], [0, e, 0, 0], [0, 0, e, 0], [0, 0, 0, 1]])
        sa = array([[1, 0, 0, 0.5], [0, se, 0, 0], [0, 0, se, 0], [0, 0, 0, 1]])
        n = a.shape[0]
        assert_array_almost_equal(dot(sa, sa), a)
        esa = sqrtm(a, disp=False, blocksize=n)[0]
        assert_array_almost_equal(dot(esa, esa), a)
        esa = sqrtm(a, disp=False, blocksize=2)[0]
        assert_array_almost_equal(dot(esa, esa), a)

    def test_sqrtm_type_preservation_and_conversion(self):
        complex_dtype_chars = ('F', 'D', 'G')
        for matrix_as_list in ([[1, 0], [0, 1]], [[1, 0], [1, 1]], [[2, 1], [1, 1]], [[2, 3], [1, 2]], [[1, 1], [1, 1]]):
            W = scipy.linalg.eigvals(matrix_as_list)
            assert_(not any((w.imag or w.real < 0 for w in W)))
            A = np.array(matrix_as_list, dtype=float)
            A_sqrtm, info = sqrtm(A, disp=False)
            assert_(A_sqrtm.dtype.char not in complex_dtype_chars)
            A = np.array(matrix_as_list, dtype=complex)
            A_sqrtm, info = sqrtm(A, disp=False)
            assert_(A_sqrtm.dtype.char in complex_dtype_chars)
            A = -np.array(matrix_as_list, dtype=float)
            A_sqrtm, info = sqrtm(A, disp=False)
            assert_(A_sqrtm.dtype.char in complex_dtype_chars)

    def test_sqrtm_type_conversion_mixed_sign_or_complex_spectrum(self):
        complex_dtype_chars = ('F', 'D', 'G')
        for matrix_as_list in ([[1, 0], [0, -1]], [[0, 1], [1, 0]], [[0, 1, 0], [0, 0, 1], [1, 0, 0]]):
            W = scipy.linalg.eigvals(matrix_as_list)
            assert_(any((w.imag or w.real < 0 for w in W)))
            A = np.array(matrix_as_list, dtype=complex)
            A_sqrtm, info = sqrtm(A, disp=False)
            assert_(A_sqrtm.dtype.char in complex_dtype_chars)
            A = np.array(matrix_as_list, dtype=float)
            A_sqrtm, info = sqrtm(A, disp=False)
            assert_(A_sqrtm.dtype.char in complex_dtype_chars)

    def test_blocksizes(self):
        np.random.seed(1234)
        for n in range(1, 8):
            A = np.random.rand(n, n) + 1j * np.random.randn(n, n)
            A_sqrtm_default, info = sqrtm(A, disp=False, blocksize=n)
            assert_allclose(A, np.linalg.matrix_power(A_sqrtm_default, 2))
            for blocksize in range(1, 10):
                A_sqrtm_new, info = sqrtm(A, disp=False, blocksize=blocksize)
                assert_allclose(A_sqrtm_default, A_sqrtm_new)

    def test_al_mohy_higham_2012_experiment_1(self):
        A = _get_al_mohy_higham_2012_experiment_1()
        A_sqrtm, info = sqrtm(A, disp=False)
        A_round_trip = A_sqrtm.dot(A_sqrtm)
        assert_allclose(A_round_trip, A, rtol=1e-05)
        assert_allclose(np.tril(A_round_trip), np.tril(A))

    def test_strict_upper_triangular(self):
        for dt in (int, float):
            A = np.array([[0, 3, 0, 0], [0, 0, 3, 0], [0, 0, 0, 3], [0, 0, 0, 0]], dtype=dt)
            A_sqrtm, info = sqrtm(A, disp=False)
            assert_(np.isnan(A_sqrtm).all())

    def test_weird_matrix(self):
        for dt in (int, float):
            A = np.array([[0, 0, 1], [0, 0, 0], [0, 1, 0]], dtype=dt)
            B = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=dt)
            assert_array_equal(B, A.dot(A))
            B_sqrtm, info = sqrtm(B, disp=False)
            assert_(np.isnan(B_sqrtm).all())

    def test_disp(self):
        np.random.seed(1234)
        A = np.random.rand(3, 3)
        B = sqrtm(A, disp=True)
        assert_allclose(B.dot(B), A)

    def test_opposite_sign_complex_eigenvalues(self):
        M = [[2j, 4], [0, -2j]]
        R = [[1 + 1j, 2], [0, 1 - 1j]]
        assert_allclose(np.dot(R, R), M, atol=1e-14)
        assert_allclose(sqrtm(M), R, atol=1e-14)

    def test_gh4866(self):
        M = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]])
        R = np.array([[sqrt(0.5), 0, 0, sqrt(0.5)], [0, 0, 0, 0], [0, 0, 0, 0], [sqrt(0.5), 0, 0, sqrt(0.5)]])
        assert_allclose(np.dot(R, R), M, atol=1e-14)
        assert_allclose(sqrtm(M), R, atol=1e-14)

    def test_gh5336(self):
        M = np.diag([2, 1, 0])
        R = np.diag([sqrt(2), 1, 0])
        assert_allclose(np.dot(R, R), M, atol=1e-14)
        assert_allclose(sqrtm(M), R, atol=1e-14)

    def test_gh7839(self):
        M = np.zeros((2, 2))
        R = np.zeros((2, 2))
        assert_allclose(np.dot(R, R), M, atol=1e-14)
        assert_allclose(sqrtm(M), R, atol=1e-14)

    def test_gh17918(self):
        M = np.empty((19, 19))
        M.fill(0.94)
        np.fill_diagonal(M, 1)
        assert np.isrealobj(sqrtm(M))

    def test_data_size_preservation_uint_in_float_out(self):
        M = np.zeros((10, 10), dtype=np.uint8)
        assert sqrtm(M).dtype == np.float16
        M = np.zeros((10, 10), dtype=np.uint16)
        assert sqrtm(M).dtype == np.float16
        M = np.zeros((10, 10), dtype=np.uint32)
        assert sqrtm(M).dtype == np.float32
        M = np.zeros((10, 10), dtype=np.uint64)
        assert sqrtm(M).dtype == np.float64

    def test_data_size_preservation_int_in_float_out(self):
        M = np.zeros((10, 10), dtype=np.int8)
        assert sqrtm(M).dtype == np.float16
        M = np.zeros((10, 10), dtype=np.int16)
        assert sqrtm(M).dtype == np.float16
        M = np.zeros((10, 10), dtype=np.int32)
        assert sqrtm(M).dtype == np.float32
        M = np.zeros((10, 10), dtype=np.int64)
        assert sqrtm(M).dtype == np.float64

    def test_data_size_preservation_int_in_comp_out(self):
        M = np.array([[2, 4], [0, -2]], dtype=np.int8)
        assert sqrtm(M).dtype == np.complex64
        M = np.array([[2, 4], [0, -2]], dtype=np.int16)
        assert sqrtm(M).dtype == np.complex64
        M = np.array([[2, 4], [0, -2]], dtype=np.int32)
        assert sqrtm(M).dtype == np.complex64
        M = np.array([[2, 4], [0, -2]], dtype=np.int64)
        assert sqrtm(M).dtype == np.complex128

    def test_data_size_preservation_float_in_float_out(self):
        M = np.zeros((10, 10), dtype=np.float16)
        assert sqrtm(M).dtype == np.float16
        M = np.zeros((10, 10), dtype=np.float32)
        assert sqrtm(M).dtype == np.float32
        M = np.zeros((10, 10), dtype=np.float64)
        assert sqrtm(M).dtype == np.float64
        if hasattr(np, 'float128'):
            M = np.zeros((10, 10), dtype=np.float128)
            assert sqrtm(M).dtype == np.float128

    def test_data_size_preservation_float_in_comp_out(self):
        M = np.array([[2, 4], [0, -2]], dtype=np.float16)
        assert sqrtm(M).dtype == np.complex64
        M = np.array([[2, 4], [0, -2]], dtype=np.float32)
        assert sqrtm(M).dtype == np.complex64
        M = np.array([[2, 4], [0, -2]], dtype=np.float64)
        assert sqrtm(M).dtype == np.complex128
        if hasattr(np, 'float128') and hasattr(np, 'complex256'):
            M = np.array([[2, 4], [0, -2]], dtype=np.float128)
            assert sqrtm(M).dtype == np.complex256

    def test_data_size_preservation_comp_in_comp_out(self):
        M = np.array([[2j, 4], [0, -2j]], dtype=np.complex64)
        assert sqrtm(M).dtype == np.complex128
        if hasattr(np, 'complex256'):
            M = np.array([[2j, 4], [0, -2j]], dtype=np.complex128)
            assert sqrtm(M).dtype == np.complex256
            M = np.array([[2j, 4], [0, -2j]], dtype=np.complex256)
            assert sqrtm(M).dtype == np.complex256