import itertools
import platform
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (eig, eigvals, lu, svd, svdvals, cholesky, qr,
from scipy.linalg.lapack import (dgbtrf, dgbtrs, zgbtrf, zgbtrs, dsbev,
from scipy.linalg._misc import norm
from scipy.linalg._decomp_qz import _select_function
from scipy.stats import ortho_group
from numpy import (array, diag, full, linalg, argsort, zeros, arange,
from scipy.linalg._testutils import assert_no_overwrite
from scipy.sparse._sputils import matrix
from scipy._lib._testutils import check_free_memory
from scipy.linalg.blas import HAS_ILP64
class TestQR:

    def test_simple(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        q, r = qr(a)
        assert_array_almost_equal(q.T @ q, eye(3))
        assert_array_almost_equal(q @ r, a)

    def test_simple_left(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        q, r = qr(a)
        c = [1, 2, 3]
        qc, r2 = qr_multiply(a, c, 'left')
        assert_array_almost_equal(q @ c, qc)
        assert_array_almost_equal(r, r2)
        qc, r2 = qr_multiply(a, eye(3), 'left')
        assert_array_almost_equal(q, qc)

    def test_simple_right(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        q, r = qr(a)
        c = [1, 2, 3]
        qc, r2 = qr_multiply(a, c)
        assert_array_almost_equal(c @ q, qc)
        assert_array_almost_equal(r, r2)
        qc, r = qr_multiply(a, eye(3))
        assert_array_almost_equal(q, qc)

    def test_simple_pivoting(self):
        a = np.asarray([[8, 2, 3], [2, 9, 3], [5, 3, 6]])
        q, r, p = qr(a, pivoting=True)
        d = abs(diag(r))
        assert_(np.all(d[1:] <= d[:-1]))
        assert_array_almost_equal(q.T @ q, eye(3))
        assert_array_almost_equal(q @ r, a[:, p])
        q2, r2 = qr(a[:, p])
        assert_array_almost_equal(q, q2)
        assert_array_almost_equal(r, r2)

    def test_simple_left_pivoting(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        q, r, jpvt = qr(a, pivoting=True)
        c = [1, 2, 3]
        qc, r, jpvt = qr_multiply(a, c, 'left', True)
        assert_array_almost_equal(q @ c, qc)

    def test_simple_right_pivoting(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        q, r, jpvt = qr(a, pivoting=True)
        c = [1, 2, 3]
        qc, r, jpvt = qr_multiply(a, c, pivoting=True)
        assert_array_almost_equal(c @ q, qc)

    def test_simple_trap(self):
        a = [[8, 2, 3], [2, 9, 3]]
        q, r = qr(a)
        assert_array_almost_equal(q.T @ q, eye(2))
        assert_array_almost_equal(q @ r, a)

    def test_simple_trap_pivoting(self):
        a = np.asarray([[8, 2, 3], [2, 9, 3]])
        q, r, p = qr(a, pivoting=True)
        d = abs(diag(r))
        assert_(np.all(d[1:] <= d[:-1]))
        assert_array_almost_equal(q.T @ q, eye(2))
        assert_array_almost_equal(q @ r, a[:, p])
        q2, r2 = qr(a[:, p])
        assert_array_almost_equal(q, q2)
        assert_array_almost_equal(r, r2)

    def test_simple_tall(self):
        a = [[8, 2], [2, 9], [5, 3]]
        q, r = qr(a)
        assert_array_almost_equal(q.T @ q, eye(3))
        assert_array_almost_equal(q @ r, a)

    def test_simple_tall_pivoting(self):
        a = np.asarray([[8, 2], [2, 9], [5, 3]])
        q, r, p = qr(a, pivoting=True)
        d = abs(diag(r))
        assert_(np.all(d[1:] <= d[:-1]))
        assert_array_almost_equal(q.T @ q, eye(3))
        assert_array_almost_equal(q @ r, a[:, p])
        q2, r2 = qr(a[:, p])
        assert_array_almost_equal(q, q2)
        assert_array_almost_equal(r, r2)

    def test_simple_tall_e(self):
        a = [[8, 2], [2, 9], [5, 3]]
        q, r = qr(a, mode='economic')
        assert_array_almost_equal(q.T @ q, eye(2))
        assert_array_almost_equal(q @ r, a)
        assert_equal(q.shape, (3, 2))
        assert_equal(r.shape, (2, 2))

    def test_simple_tall_e_pivoting(self):
        a = np.asarray([[8, 2], [2, 9], [5, 3]])
        q, r, p = qr(a, pivoting=True, mode='economic')
        d = abs(diag(r))
        assert_(np.all(d[1:] <= d[:-1]))
        assert_array_almost_equal(q.T @ q, eye(2))
        assert_array_almost_equal(q @ r, a[:, p])
        q2, r2 = qr(a[:, p], mode='economic')
        assert_array_almost_equal(q, q2)
        assert_array_almost_equal(r, r2)

    def test_simple_tall_left(self):
        a = [[8, 2], [2, 9], [5, 3]]
        q, r = qr(a, mode='economic')
        c = [1, 2]
        qc, r2 = qr_multiply(a, c, 'left')
        assert_array_almost_equal(q @ c, qc)
        assert_array_almost_equal(r, r2)
        c = array([1, 2, 0])
        qc, r2 = qr_multiply(a, c, 'left', overwrite_c=True)
        assert_array_almost_equal(q @ c[:2], qc)
        qc, r = qr_multiply(a, eye(2), 'left')
        assert_array_almost_equal(qc, q)

    def test_simple_tall_left_pivoting(self):
        a = [[8, 2], [2, 9], [5, 3]]
        q, r, jpvt = qr(a, mode='economic', pivoting=True)
        c = [1, 2]
        qc, r, kpvt = qr_multiply(a, c, 'left', True)
        assert_array_equal(jpvt, kpvt)
        assert_array_almost_equal(q @ c, qc)
        qc, r, jpvt = qr_multiply(a, eye(2), 'left', True)
        assert_array_almost_equal(qc, q)

    def test_simple_tall_right(self):
        a = [[8, 2], [2, 9], [5, 3]]
        q, r = qr(a, mode='economic')
        c = [1, 2, 3]
        cq, r2 = qr_multiply(a, c)
        assert_array_almost_equal(c @ q, cq)
        assert_array_almost_equal(r, r2)
        cq, r = qr_multiply(a, eye(3))
        assert_array_almost_equal(cq, q)

    def test_simple_tall_right_pivoting(self):
        a = [[8, 2], [2, 9], [5, 3]]
        q, r, jpvt = qr(a, pivoting=True, mode='economic')
        c = [1, 2, 3]
        cq, r, jpvt = qr_multiply(a, c, pivoting=True)
        assert_array_almost_equal(c @ q, cq)
        cq, r, jpvt = qr_multiply(a, eye(3), pivoting=True)
        assert_array_almost_equal(cq, q)

    def test_simple_fat(self):
        a = [[8, 2, 5], [2, 9, 3]]
        q, r = qr(a)
        assert_array_almost_equal(q.T @ q, eye(2))
        assert_array_almost_equal(q @ r, a)
        assert_equal(q.shape, (2, 2))
        assert_equal(r.shape, (2, 3))

    def test_simple_fat_pivoting(self):
        a = np.asarray([[8, 2, 5], [2, 9, 3]])
        q, r, p = qr(a, pivoting=True)
        d = abs(diag(r))
        assert_(np.all(d[1:] <= d[:-1]))
        assert_array_almost_equal(q.T @ q, eye(2))
        assert_array_almost_equal(q @ r, a[:, p])
        assert_equal(q.shape, (2, 2))
        assert_equal(r.shape, (2, 3))
        q2, r2 = qr(a[:, p])
        assert_array_almost_equal(q, q2)
        assert_array_almost_equal(r, r2)

    def test_simple_fat_e(self):
        a = [[8, 2, 3], [2, 9, 5]]
        q, r = qr(a, mode='economic')
        assert_array_almost_equal(q.T @ q, eye(2))
        assert_array_almost_equal(q @ r, a)
        assert_equal(q.shape, (2, 2))
        assert_equal(r.shape, (2, 3))

    def test_simple_fat_e_pivoting(self):
        a = np.asarray([[8, 2, 3], [2, 9, 5]])
        q, r, p = qr(a, pivoting=True, mode='economic')
        d = abs(diag(r))
        assert_(np.all(d[1:] <= d[:-1]))
        assert_array_almost_equal(q.T @ q, eye(2))
        assert_array_almost_equal(q @ r, a[:, p])
        assert_equal(q.shape, (2, 2))
        assert_equal(r.shape, (2, 3))
        q2, r2 = qr(a[:, p], mode='economic')
        assert_array_almost_equal(q, q2)
        assert_array_almost_equal(r, r2)

    def test_simple_fat_left(self):
        a = [[8, 2, 3], [2, 9, 5]]
        q, r = qr(a, mode='economic')
        c = [1, 2]
        qc, r2 = qr_multiply(a, c, 'left')
        assert_array_almost_equal(q @ c, qc)
        assert_array_almost_equal(r, r2)
        qc, r = qr_multiply(a, eye(2), 'left')
        assert_array_almost_equal(qc, q)

    def test_simple_fat_left_pivoting(self):
        a = [[8, 2, 3], [2, 9, 5]]
        q, r, jpvt = qr(a, mode='economic', pivoting=True)
        c = [1, 2]
        qc, r, jpvt = qr_multiply(a, c, 'left', True)
        assert_array_almost_equal(q @ c, qc)
        qc, r, jpvt = qr_multiply(a, eye(2), 'left', True)
        assert_array_almost_equal(qc, q)

    def test_simple_fat_right(self):
        a = [[8, 2, 3], [2, 9, 5]]
        q, r = qr(a, mode='economic')
        c = [1, 2]
        cq, r2 = qr_multiply(a, c)
        assert_array_almost_equal(c @ q, cq)
        assert_array_almost_equal(r, r2)
        cq, r = qr_multiply(a, eye(2))
        assert_array_almost_equal(cq, q)

    def test_simple_fat_right_pivoting(self):
        a = [[8, 2, 3], [2, 9, 5]]
        q, r, jpvt = qr(a, pivoting=True, mode='economic')
        c = [1, 2]
        cq, r, jpvt = qr_multiply(a, c, pivoting=True)
        assert_array_almost_equal(c @ q, cq)
        cq, r, jpvt = qr_multiply(a, eye(2), pivoting=True)
        assert_array_almost_equal(cq, q)

    def test_simple_complex(self):
        a = [[3, 3 + 4j, 5], [5, 2, 2 + 7j], [3, 2, 7]]
        q, r = qr(a)
        assert_array_almost_equal(q.conj().T @ q, eye(3))
        assert_array_almost_equal(q @ r, a)

    def test_simple_complex_left(self):
        a = [[3, 3 + 4j, 5], [5, 2, 2 + 7j], [3, 2, 7]]
        q, r = qr(a)
        c = [1, 2, 3 + 4j]
        qc, r = qr_multiply(a, c, 'left')
        assert_array_almost_equal(q @ c, qc)
        qc, r = qr_multiply(a, eye(3), 'left')
        assert_array_almost_equal(q, qc)

    def test_simple_complex_right(self):
        a = [[3, 3 + 4j, 5], [5, 2, 2 + 7j], [3, 2, 7]]
        q, r = qr(a)
        c = [1, 2, 3 + 4j]
        qc, r = qr_multiply(a, c)
        assert_array_almost_equal(c @ q, qc)
        qc, r = qr_multiply(a, eye(3))
        assert_array_almost_equal(q, qc)

    def test_simple_tall_complex_left(self):
        a = [[8, 2 + 3j], [2, 9], [5 + 7j, 3]]
        q, r = qr(a, mode='economic')
        c = [1, 2 + 2j]
        qc, r2 = qr_multiply(a, c, 'left')
        assert_array_almost_equal(q @ c, qc)
        assert_array_almost_equal(r, r2)
        c = array([1, 2, 0])
        qc, r2 = qr_multiply(a, c, 'left', overwrite_c=True)
        assert_array_almost_equal(q @ c[:2], qc)
        qc, r = qr_multiply(a, eye(2), 'left')
        assert_array_almost_equal(qc, q)

    def test_simple_complex_left_conjugate(self):
        a = [[3, 3 + 4j, 5], [5, 2, 2 + 7j], [3, 2, 7]]
        q, r = qr(a)
        c = [1, 2, 3 + 4j]
        qc, r = qr_multiply(a, c, 'left', conjugate=True)
        assert_array_almost_equal(q.conj() @ c, qc)

    def test_simple_complex_tall_left_conjugate(self):
        a = [[3, 3 + 4j], [5, 2 + 2j], [3, 2]]
        q, r = qr(a, mode='economic')
        c = [1, 3 + 4j]
        qc, r = qr_multiply(a, c, 'left', conjugate=True)
        assert_array_almost_equal(q.conj() @ c, qc)

    def test_simple_complex_right_conjugate(self):
        a = [[3, 3 + 4j, 5], [5, 2, 2 + 7j], [3, 2, 7]]
        q, r = qr(a)
        c = np.array([1, 2, 3 + 4j])
        qc, r = qr_multiply(a, c, conjugate=True)
        assert_array_almost_equal(c @ q.conj(), qc)

    def test_simple_complex_pivoting(self):
        a = array([[3, 3 + 4j, 5], [5, 2, 2 + 7j], [3, 2, 7]])
        q, r, p = qr(a, pivoting=True)
        d = abs(diag(r))
        assert_(np.all(d[1:] <= d[:-1]))
        assert_array_almost_equal(q.conj().T @ q, eye(3))
        assert_array_almost_equal(q @ r, a[:, p])
        q2, r2 = qr(a[:, p])
        assert_array_almost_equal(q, q2)
        assert_array_almost_equal(r, r2)

    def test_simple_complex_left_pivoting(self):
        a = array([[3, 3 + 4j, 5], [5, 2, 2 + 7j], [3, 2, 7]])
        q, r, jpvt = qr(a, pivoting=True)
        c = [1, 2, 3 + 4j]
        qc, r, jpvt = qr_multiply(a, c, 'left', True)
        assert_array_almost_equal(q @ c, qc)

    def test_simple_complex_right_pivoting(self):
        a = array([[3, 3 + 4j, 5], [5, 2, 2 + 7j], [3, 2, 7]])
        q, r, jpvt = qr(a, pivoting=True)
        c = [1, 2, 3 + 4j]
        qc, r, jpvt = qr_multiply(a, c, pivoting=True)
        assert_array_almost_equal(c @ q, qc)

    def test_random(self):
        rng = np.random.RandomState(1234)
        n = 20
        for k in range(2):
            a = rng.random([n, n])
            q, r = qr(a)
            assert_array_almost_equal(q.T @ q, eye(n))
            assert_array_almost_equal(q @ r, a)

    def test_random_left(self):
        rng = np.random.RandomState(1234)
        n = 20
        for k in range(2):
            a = rng.random([n, n])
            q, r = qr(a)
            c = rng.random([n])
            qc, r = qr_multiply(a, c, 'left')
            assert_array_almost_equal(q @ c, qc)
            qc, r = qr_multiply(a, eye(n), 'left')
            assert_array_almost_equal(q, qc)

    def test_random_right(self):
        rng = np.random.RandomState(1234)
        n = 20
        for k in range(2):
            a = rng.random([n, n])
            q, r = qr(a)
            c = rng.random([n])
            cq, r = qr_multiply(a, c)
            assert_array_almost_equal(c @ q, cq)
            cq, r = qr_multiply(a, eye(n))
            assert_array_almost_equal(q, cq)

    def test_random_pivoting(self):
        rng = np.random.RandomState(1234)
        n = 20
        for k in range(2):
            a = rng.random([n, n])
            q, r, p = qr(a, pivoting=True)
            d = abs(diag(r))
            assert_(np.all(d[1:] <= d[:-1]))
            assert_array_almost_equal(q.T @ q, eye(n))
            assert_array_almost_equal(q @ r, a[:, p])
            q2, r2 = qr(a[:, p])
            assert_array_almost_equal(q, q2)
            assert_array_almost_equal(r, r2)

    def test_random_tall(self):
        rng = np.random.RandomState(1234)
        m = 200
        n = 100
        for k in range(2):
            a = rng.random([m, n])
            q, r = qr(a)
            assert_array_almost_equal(q.T @ q, eye(m))
            assert_array_almost_equal(q @ r, a)

    def test_random_tall_left(self):
        rng = np.random.RandomState(1234)
        m = 200
        n = 100
        for k in range(2):
            a = rng.random([m, n])
            q, r = qr(a, mode='economic')
            c = rng.random([n])
            qc, r = qr_multiply(a, c, 'left')
            assert_array_almost_equal(q @ c, qc)
            qc, r = qr_multiply(a, eye(n), 'left')
            assert_array_almost_equal(qc, q)

    def test_random_tall_right(self):
        rng = np.random.RandomState(1234)
        m = 200
        n = 100
        for k in range(2):
            a = rng.random([m, n])
            q, r = qr(a, mode='economic')
            c = rng.random([m])
            cq, r = qr_multiply(a, c)
            assert_array_almost_equal(c @ q, cq)
            cq, r = qr_multiply(a, eye(m))
            assert_array_almost_equal(cq, q)

    def test_random_tall_pivoting(self):
        rng = np.random.RandomState(1234)
        m = 200
        n = 100
        for k in range(2):
            a = rng.random([m, n])
            q, r, p = qr(a, pivoting=True)
            d = abs(diag(r))
            assert_(np.all(d[1:] <= d[:-1]))
            assert_array_almost_equal(q.T @ q, eye(m))
            assert_array_almost_equal(q @ r, a[:, p])
            q2, r2 = qr(a[:, p])
            assert_array_almost_equal(q, q2)
            assert_array_almost_equal(r, r2)

    def test_random_tall_e(self):
        rng = np.random.RandomState(1234)
        m = 200
        n = 100
        for k in range(2):
            a = rng.random([m, n])
            q, r = qr(a, mode='economic')
            assert_array_almost_equal(q.T @ q, eye(n))
            assert_array_almost_equal(q @ r, a)
            assert_equal(q.shape, (m, n))
            assert_equal(r.shape, (n, n))

    def test_random_tall_e_pivoting(self):
        rng = np.random.RandomState(1234)
        m = 200
        n = 100
        for k in range(2):
            a = rng.random([m, n])
            q, r, p = qr(a, pivoting=True, mode='economic')
            d = abs(diag(r))
            assert_(np.all(d[1:] <= d[:-1]))
            assert_array_almost_equal(q.T @ q, eye(n))
            assert_array_almost_equal(q @ r, a[:, p])
            assert_equal(q.shape, (m, n))
            assert_equal(r.shape, (n, n))
            q2, r2 = qr(a[:, p], mode='economic')
            assert_array_almost_equal(q, q2)
            assert_array_almost_equal(r, r2)

    def test_random_trap(self):
        rng = np.random.RandomState(1234)
        m = 100
        n = 200
        for k in range(2):
            a = rng.random([m, n])
            q, r = qr(a)
            assert_array_almost_equal(q.T @ q, eye(m))
            assert_array_almost_equal(q @ r, a)

    def test_random_trap_pivoting(self):
        rng = np.random.RandomState(1234)
        m = 100
        n = 200
        for k in range(2):
            a = rng.random([m, n])
            q, r, p = qr(a, pivoting=True)
            d = abs(diag(r))
            assert_(np.all(d[1:] <= d[:-1]))
            assert_array_almost_equal(q.T @ q, eye(m))
            assert_array_almost_equal(q @ r, a[:, p])
            q2, r2 = qr(a[:, p])
            assert_array_almost_equal(q, q2)
            assert_array_almost_equal(r, r2)

    def test_random_complex(self):
        rng = np.random.RandomState(1234)
        n = 20
        for k in range(2):
            a = rng.random([n, n]) + 1j * rng.random([n, n])
            q, r = qr(a)
            assert_array_almost_equal(q.conj().T @ q, eye(n))
            assert_array_almost_equal(q @ r, a)

    def test_random_complex_left(self):
        rng = np.random.RandomState(1234)
        n = 20
        for k in range(2):
            a = rng.random([n, n]) + 1j * rng.random([n, n])
            q, r = qr(a)
            c = rng.random([n]) + 1j * rng.random([n])
            qc, r = qr_multiply(a, c, 'left')
            assert_array_almost_equal(q @ c, qc)
            qc, r = qr_multiply(a, eye(n), 'left')
            assert_array_almost_equal(q, qc)

    def test_random_complex_right(self):
        rng = np.random.RandomState(1234)
        n = 20
        for k in range(2):
            a = rng.random([n, n]) + 1j * rng.random([n, n])
            q, r = qr(a)
            c = rng.random([n]) + 1j * rng.random([n])
            cq, r = qr_multiply(a, c)
            assert_array_almost_equal(c @ q, cq)
            cq, r = qr_multiply(a, eye(n))
            assert_array_almost_equal(q, cq)

    def test_random_complex_pivoting(self):
        rng = np.random.RandomState(1234)
        n = 20
        for k in range(2):
            a = rng.random([n, n]) + 1j * rng.random([n, n])
            q, r, p = qr(a, pivoting=True)
            d = abs(diag(r))
            assert_(np.all(d[1:] <= d[:-1]))
            assert_array_almost_equal(q.conj().T @ q, eye(n))
            assert_array_almost_equal(q @ r, a[:, p])
            q2, r2 = qr(a[:, p])
            assert_array_almost_equal(q, q2)
            assert_array_almost_equal(r, r2)

    def test_check_finite(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        q, r = qr(a, check_finite=False)
        assert_array_almost_equal(q.T @ q, eye(3))
        assert_array_almost_equal(q @ r, a)

    def test_lwork(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        q, r = qr(a, lwork=None)
        q2, r2 = qr(a, lwork=3)
        assert_array_almost_equal(q2, q)
        assert_array_almost_equal(r2, r)
        q3, r3 = qr(a, lwork=10)
        assert_array_almost_equal(q3, q)
        assert_array_almost_equal(r3, r)
        q4, r4 = qr(a, lwork=-1)
        assert_array_almost_equal(q4, q)
        assert_array_almost_equal(r4, r)
        assert_raises(Exception, qr, (a,), {'lwork': 0})
        assert_raises(Exception, qr, (a,), {'lwork': 2})