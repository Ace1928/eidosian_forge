import itertools
import warnings
import numpy as np
from numpy import (arange, array, dot, zeros, identity, conjugate, transpose,
from numpy.random import random
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (solve, inv, det, lstsq, pinv, pinvh, norm,
from scipy.linalg._testutils import assert_no_overwrite
from scipy._lib._testutils import check_free_memory, IS_MUSL
from scipy.linalg.blas import HAS_ILP64
from scipy._lib.deprecation import _NoValue
class TestSolve:

    def setup_method(self):
        np.random.seed(1234)

    def test_20Feb04_bug(self):
        a = [[1, 1], [1.0, 0]]
        x0 = solve(a, [1, 0j])
        assert_array_almost_equal(dot(a, x0), [1, 0])
        a = [[1, 1], [1.2, 0]]
        b = [1, 0j]
        x0 = solve(a, b)
        assert_array_almost_equal(dot(a, x0), [1, 0])

    def test_simple(self):
        a = [[1, 20], [-30, 4]]
        for b in ([[1, 0], [0, 1]], [1, 0], [[2, 1], [-30, 4]]):
            x = solve(a, b)
            assert_array_almost_equal(dot(a, x), b)

    def test_simple_complex(self):
        a = array([[5, 2], [2j, 4]], 'D')
        for b in ([1j, 0], [[1j, 1j], [0, 2]], [1, 0j], array([1, 0], 'D')):
            x = solve(a, b)
            assert_array_almost_equal(dot(a, x), b)

    def test_simple_pos(self):
        a = [[2, 3], [3, 5]]
        for lower in [0, 1]:
            for b in ([[1, 0], [0, 1]], [1, 0]):
                x = solve(a, b, assume_a='pos', lower=lower)
                assert_array_almost_equal(dot(a, x), b)

    def test_simple_pos_complexb(self):
        a = [[5, 2], [2, 4]]
        for b in ([1j, 0], [[1j, 1j], [0, 2]]):
            x = solve(a, b, assume_a='pos')
            assert_array_almost_equal(dot(a, x), b)

    def test_simple_sym(self):
        a = [[2, 3], [3, -5]]
        for lower in [0, 1]:
            for b in ([[1, 0], [0, 1]], [1, 0]):
                x = solve(a, b, assume_a='sym', lower=lower)
                assert_array_almost_equal(dot(a, x), b)

    def test_simple_sym_complexb(self):
        a = [[5, 2], [2, -4]]
        for b in ([1j, 0], [[1j, 1j], [0, 2]]):
            x = solve(a, b, assume_a='sym')
            assert_array_almost_equal(dot(a, x), b)

    def test_simple_sym_complex(self):
        a = [[5, 2 + 1j], [2 + 1j, -4]]
        for b in ([1j, 0], [1, 0], [[1j, 1j], [0, 2]]):
            x = solve(a, b, assume_a='sym')
            assert_array_almost_equal(dot(a, x), b)

    def test_simple_her_actuallysym(self):
        a = [[2, 3], [3, -5]]
        for lower in [0, 1]:
            for b in ([[1, 0], [0, 1]], [1, 0], [1j, 0]):
                x = solve(a, b, assume_a='her', lower=lower)
                assert_array_almost_equal(dot(a, x), b)

    def test_simple_her(self):
        a = [[5, 2 + 1j], [2 - 1j, -4]]
        for b in ([1j, 0], [1, 0], [[1j, 1j], [0, 2]]):
            x = solve(a, b, assume_a='her')
            assert_array_almost_equal(dot(a, x), b)

    def test_nils_20Feb04(self):
        n = 2
        A = random([n, n]) + random([n, n]) * 1j
        X = zeros((n, n), 'D')
        Ainv = inv(A)
        R = identity(n) + identity(n) * 0j
        for i in arange(0, n):
            r = R[:, i]
            X[:, i] = solve(A, r)
        assert_array_almost_equal(X, Ainv)

    def test_random(self):
        n = 20
        a = random([n, n])
        for i in range(n):
            a[i, i] = 20 * (0.1 + a[i, i])
        for i in range(4):
            b = random([n, 3])
            x = solve(a, b)
            assert_array_almost_equal(dot(a, x), b)

    def test_random_complex(self):
        n = 20
        a = random([n, n]) + 1j * random([n, n])
        for i in range(n):
            a[i, i] = 20 * (0.1 + a[i, i])
        for i in range(2):
            b = random([n, 3])
            x = solve(a, b)
            assert_array_almost_equal(dot(a, x), b)

    def test_random_sym(self):
        n = 20
        a = random([n, n])
        for i in range(n):
            a[i, i] = abs(20 * (0.1 + a[i, i]))
            for j in range(i):
                a[i, j] = a[j, i]
        for i in range(4):
            b = random([n])
            x = solve(a, b, assume_a='pos')
            assert_array_almost_equal(dot(a, x), b)

    def test_random_sym_complex(self):
        n = 20
        a = random([n, n])
        a = a + 1j * random([n, n])
        for i in range(n):
            a[i, i] = abs(20 * (0.1 + a[i, i]))
            for j in range(i):
                a[i, j] = conjugate(a[j, i])
        b = random([n]) + 2j * random([n])
        for i in range(2):
            x = solve(a, b, assume_a='pos')
            assert_array_almost_equal(dot(a, x), b)

    def test_check_finite(self):
        a = [[1, 20], [-30, 4]]
        for b in ([[1, 0], [0, 1]], [1, 0], [[2, 1], [-30, 4]]):
            x = solve(a, b, check_finite=False)
            assert_array_almost_equal(dot(a, x), b)

    def test_scalar_a_and_1D_b(self):
        a = 1
        b = [1, 2, 3]
        x = solve(a, b)
        assert_array_almost_equal(x.ravel(), b)
        assert_(x.shape == (3,), 'Scalar_a_1D_b test returned wrong shape')

    def test_simple2(self):
        a = np.array([[1.8, 2.88, 2.05, -0.89], [525.0, -295.0, -95.0, -380.0], [1.58, -2.69, -2.9, -1.04], [-1.11, -0.66, -0.59, 0.8]])
        b = np.array([[9.52, 18.47], [2435.0, 225.0], [0.77, -13.28], [-6.22, -6.21]])
        x = solve(a, b)
        assert_array_almost_equal(x, np.array([[1.0, -1, 3, -5], [3, 2, 4, 1]]).T)

    def test_simple_complex2(self):
        a = np.array([[-1.34 + 2.55j, 0.28 + 3.17j, -6.39 - 2.2j, 0.72 - 0.92j], [-1.7 - 14.1j, 33.1 - 1.5j, -1.5 + 13.4j, 12.9 + 13.8j], [-3.29 - 2.39j, -1.91 + 4.42j, -0.14 - 1.35j, 1.72 + 1.35j], [2.41 + 0.39j, -0.56 + 1.47j, -0.83 - 0.69j, -1.96 + 0.67j]])
        b = np.array([[26.26 + 51.78j, 31.32 - 6.7j], [64.3 - 86.8j, 158.6 - 14.2j], [-5.75 + 25.31j, -2.15 + 30.19j], [1.16 + 2.57j, -2.56 + 7.55j]])
        x = solve(a, b)
        assert_array_almost_equal(x, np.array([[1 + 1j, -1 - 2j], [2 - 3j, 5 + 1j], [-4 - 5j, -3 + 4j], [6j, 2 - 3j]]))

    def test_hermitian(self):
        a = np.array([[-1.84, 0.11 - 0.11j, -1.78 - 1.18j, 3.91 - 1.5j], [0, -4.63, -1.84 + 0.03j, 2.21 + 0.21j], [0, 0, -8.87, 1.58 - 0.9j], [0, 0, 0, -1.36]])
        b = np.array([[2.98 - 10.18j, 28.68 - 39.89j], [-9.58 + 3.88j, -24.79 - 8.4j], [-0.77 - 16.05j, 4.23 - 70.02j], [7.79 + 5.48j, -35.39 + 18.01j]])
        res = np.array([[2.0 + 1j, -8 + 6j], [3.0 - 2j, 7 - 2j], [-1 + 2j, -1 + 5j], [1.0 - 1j, 3 - 4j]])
        x = solve(a, b, assume_a='her')
        assert_array_almost_equal(x, res)
        x = solve(a.conj().T, b, assume_a='her', lower=True)
        assert_array_almost_equal(x, res)

    def test_pos_and_sym(self):
        A = np.arange(1, 10).reshape(3, 3)
        x = solve(np.tril(A) / 9, np.ones(3), assume_a='pos')
        assert_array_almost_equal(x, [9.0, 1.8, 1.0])
        x = solve(np.tril(A) / 9, np.ones(3), assume_a='sym')
        assert_array_almost_equal(x, [9.0, 1.8, 1.0])

    def test_singularity(self):
        a = np.array([[1, 0, 0, 0, 0, 0, 1, 0, 1], [1, 1, 1, 0, 0, 0, 1, 0, 1], [0, 1, 1, 0, 0, 0, 1, 0, 1], [1, 0, 1, 1, 1, 1, 0, 0, 0], [1, 0, 1, 1, 1, 1, 0, 0, 0], [1, 0, 1, 1, 1, 1, 0, 0, 0], [1, 0, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]])
        b = np.arange(9)[:, None]
        assert_raises(LinAlgError, solve, a, b)

    def test_ill_condition_warning(self):
        a = np.array([[1, 1], [1 + 1e-16, 1 - 1e-16]])
        b = np.ones(2)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert_raises(LinAlgWarning, solve, a, b)

    def test_empty_rhs(self):
        a = np.eye(2)
        b = [[], []]
        x = solve(a, b)
        assert_(x.size == 0, 'Returned array is not empty')
        assert_(x.shape == (2, 0), 'Returned empty array shape is wrong')

    def test_multiple_rhs(self):
        a = np.eye(2)
        b = np.random.rand(2, 3, 4)
        x = solve(a, b)
        assert_array_almost_equal(x, b)

    def test_transposed_keyword(self):
        A = np.arange(9).reshape(3, 3) + 1
        x = solve(np.tril(A) / 9, np.ones(3), transposed=True)
        assert_array_almost_equal(x, [1.2, 0.2, 1])
        x = solve(np.tril(A) / 9, np.ones(3), transposed=False)
        assert_array_almost_equal(x, [9, -5.4, -1.2])

    def test_transposed_notimplemented(self):
        a = np.eye(3).astype(complex)
        with assert_raises(NotImplementedError):
            solve(a, a, transposed=True)

    def test_nonsquare_a(self):
        assert_raises(ValueError, solve, [1, 2], 1)

    def test_size_mismatch_with_1D_b(self):
        assert_array_almost_equal(solve(np.eye(3), np.ones(3)), np.ones(3))
        assert_raises(ValueError, solve, np.eye(3), np.ones(4))

    def test_assume_a_keyword(self):
        assert_raises(ValueError, solve, 1, 1, assume_a='zxcv')

    @pytest.mark.skip(reason='Failure on OS X (gh-7500), crash on Windows (gh-8064)')
    def test_all_type_size_routine_combinations(self):
        sizes = [10, 100]
        assume_as = ['gen', 'sym', 'pos', 'her']
        dtypes = [np.float32, np.float64, np.complex64, np.complex128]
        for size, assume_a, dtype in itertools.product(sizes, assume_as, dtypes):
            is_complex = dtype in (np.complex64, np.complex128)
            if assume_a == 'her' and (not is_complex):
                continue
            err_msg = f'Failed for size: {size}, assume_a: {assume_a},dtype: {dtype}'
            a = np.random.randn(size, size).astype(dtype)
            b = np.random.randn(size).astype(dtype)
            if is_complex:
                a = a + (1j * np.random.randn(size, size)).astype(dtype)
            if assume_a == 'sym':
                a = a + a.T
            elif assume_a == 'her':
                a = a + a.T.conj()
            elif assume_a == 'pos':
                a = a.conj().T.dot(a) + 0.1 * np.eye(size)
            tol = 1e-12 if dtype in (np.float64, np.complex128) else 1e-06
            if assume_a in ['gen', 'sym', 'her']:
                if dtype in (np.float32, np.complex64):
                    tol *= 10
            x = solve(a, b, assume_a=assume_a)
            assert_allclose(a.dot(x), b, atol=tol * size, rtol=tol * size, err_msg=err_msg)
            if assume_a == 'sym' and dtype not in (np.complex64, np.complex128):
                x = solve(a, b, assume_a=assume_a, transposed=True)
                assert_allclose(a.dot(x), b, atol=tol * size, rtol=tol * size, err_msg=err_msg)