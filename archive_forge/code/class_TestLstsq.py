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
class TestLstsq:
    lapack_drivers = ('gelsd', 'gelss', 'gelsy', None)

    def test_simple_exact(self):
        for dtype in REAL_DTYPES:
            a = np.array([[1, 20], [-30, 4]], dtype=dtype)
            for lapack_driver in TestLstsq.lapack_drivers:
                for overwrite in (True, False):
                    for bt in (((1, 0), (0, 1)), (1, 0), ((2, 1), (-30, 4))):
                        a1 = a.copy()
                        b = np.array(bt, dtype=dtype)
                        b1 = b.copy()
                        out = lstsq(a1, b1, lapack_driver=lapack_driver, overwrite_a=overwrite, overwrite_b=overwrite)
                        x = out[0]
                        r = out[2]
                        assert_(r == 2, 'expected efficient rank 2, got %s' % r)
                        assert_allclose(dot(a, x), b, atol=25 * _eps_cast(a1.dtype), rtol=25 * _eps_cast(a1.dtype), err_msg='driver: %s' % lapack_driver)

    def test_simple_overdet(self):
        for dtype in REAL_DTYPES:
            a = np.array([[1, 2], [4, 5], [3, 4]], dtype=dtype)
            b = np.array([1, 2, 3], dtype=dtype)
            for lapack_driver in TestLstsq.lapack_drivers:
                for overwrite in (True, False):
                    a1 = a.copy()
                    b1 = b.copy()
                    out = lstsq(a1, b1, lapack_driver=lapack_driver, overwrite_a=overwrite, overwrite_b=overwrite)
                    x = out[0]
                    if lapack_driver == 'gelsy':
                        residuals = np.sum((b - a.dot(x)) ** 2)
                    else:
                        residuals = out[1]
                    r = out[2]
                    assert_(r == 2, 'expected efficient rank 2, got %s' % r)
                    assert_allclose(abs((dot(a, x) - b) ** 2).sum(axis=0), residuals, rtol=25 * _eps_cast(a1.dtype), atol=25 * _eps_cast(a1.dtype), err_msg='driver: %s' % lapack_driver)
                    assert_allclose(x, (-0.428571428571429, 0.85714285714285), rtol=25 * _eps_cast(a1.dtype), atol=25 * _eps_cast(a1.dtype), err_msg='driver: %s' % lapack_driver)

    def test_simple_overdet_complex(self):
        for dtype in COMPLEX_DTYPES:
            a = np.array([[1 + 2j, 2], [4, 5], [3, 4]], dtype=dtype)
            b = np.array([1, 2 + 4j, 3], dtype=dtype)
            for lapack_driver in TestLstsq.lapack_drivers:
                for overwrite in (True, False):
                    a1 = a.copy()
                    b1 = b.copy()
                    out = lstsq(a1, b1, lapack_driver=lapack_driver, overwrite_a=overwrite, overwrite_b=overwrite)
                    x = out[0]
                    if lapack_driver == 'gelsy':
                        res = b - a.dot(x)
                        residuals = np.sum(res * res.conj())
                    else:
                        residuals = out[1]
                    r = out[2]
                    assert_(r == 2, 'expected efficient rank 2, got %s' % r)
                    assert_allclose(abs((dot(a, x) - b) ** 2).sum(axis=0), residuals, rtol=25 * _eps_cast(a1.dtype), atol=25 * _eps_cast(a1.dtype), err_msg='driver: %s' % lapack_driver)
                    assert_allclose(x, (-0.4831460674157303 + 0.258426966292135j, 0.921348314606741 + 0.292134831460674j), rtol=25 * _eps_cast(a1.dtype), atol=25 * _eps_cast(a1.dtype), err_msg='driver: %s' % lapack_driver)

    def test_simple_underdet(self):
        for dtype in REAL_DTYPES:
            a = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
            b = np.array([1, 2], dtype=dtype)
            for lapack_driver in TestLstsq.lapack_drivers:
                for overwrite in (True, False):
                    a1 = a.copy()
                    b1 = b.copy()
                    out = lstsq(a1, b1, lapack_driver=lapack_driver, overwrite_a=overwrite, overwrite_b=overwrite)
                    x = out[0]
                    r = out[2]
                    assert_(r == 2, 'expected efficient rank 2, got %s' % r)
                    assert_allclose(x, (-0.055555555555555, 0.111111111111111, 0.277777777777777), rtol=25 * _eps_cast(a1.dtype), atol=25 * _eps_cast(a1.dtype), err_msg='driver: %s' % lapack_driver)

    def test_random_exact(self):
        rng = np.random.RandomState(1234)
        for dtype in REAL_DTYPES:
            for n in (20, 200):
                for lapack_driver in TestLstsq.lapack_drivers:
                    for overwrite in (True, False):
                        a = np.asarray(rng.random([n, n]), dtype=dtype)
                        for i in range(n):
                            a[i, i] = 20 * (0.1 + a[i, i])
                        for i in range(4):
                            b = np.asarray(rng.random([n, 3]), dtype=dtype)
                            a1 = a.copy()
                            b1 = b.copy()
                            out = lstsq(a1, b1, lapack_driver=lapack_driver, overwrite_a=overwrite, overwrite_b=overwrite)
                            x = out[0]
                            r = out[2]
                            assert_(r == n, f'expected efficient rank {n}, got {r}')
                            if dtype is np.float32:
                                assert_allclose(dot(a, x), b, rtol=500 * _eps_cast(a1.dtype), atol=500 * _eps_cast(a1.dtype), err_msg='driver: %s' % lapack_driver)
                            else:
                                assert_allclose(dot(a, x), b, rtol=1000 * _eps_cast(a1.dtype), atol=1000 * _eps_cast(a1.dtype), err_msg='driver: %s' % lapack_driver)

    @pytest.mark.skipif(IS_MUSL, reason='may segfault on Alpine, see gh-17630')
    def test_random_complex_exact(self):
        rng = np.random.RandomState(1234)
        for dtype in COMPLEX_DTYPES:
            for n in (20, 200):
                for lapack_driver in TestLstsq.lapack_drivers:
                    for overwrite in (True, False):
                        a = np.asarray(rng.random([n, n]) + 1j * rng.random([n, n]), dtype=dtype)
                        for i in range(n):
                            a[i, i] = 20 * (0.1 + a[i, i])
                        for i in range(2):
                            b = np.asarray(rng.random([n, 3]), dtype=dtype)
                            a1 = a.copy()
                            b1 = b.copy()
                            out = lstsq(a1, b1, lapack_driver=lapack_driver, overwrite_a=overwrite, overwrite_b=overwrite)
                            x = out[0]
                            r = out[2]
                            assert_(r == n, f'expected efficient rank {n}, got {r}')
                            if dtype is np.complex64:
                                assert_allclose(dot(a, x), b, rtol=400 * _eps_cast(a1.dtype), atol=400 * _eps_cast(a1.dtype), err_msg='driver: %s' % lapack_driver)
                            else:
                                assert_allclose(dot(a, x), b, rtol=1000 * _eps_cast(a1.dtype), atol=1000 * _eps_cast(a1.dtype), err_msg='driver: %s' % lapack_driver)

    def test_random_overdet(self):
        rng = np.random.RandomState(1234)
        for dtype in REAL_DTYPES:
            for n, m in ((20, 15), (200, 2)):
                for lapack_driver in TestLstsq.lapack_drivers:
                    for overwrite in (True, False):
                        a = np.asarray(rng.random([n, m]), dtype=dtype)
                        for i in range(m):
                            a[i, i] = 20 * (0.1 + a[i, i])
                        for i in range(4):
                            b = np.asarray(rng.random([n, 3]), dtype=dtype)
                            a1 = a.copy()
                            b1 = b.copy()
                            out = lstsq(a1, b1, lapack_driver=lapack_driver, overwrite_a=overwrite, overwrite_b=overwrite)
                            x = out[0]
                            r = out[2]
                            assert_(r == m, f'expected efficient rank {m}, got {r}')
                            assert_allclose(x, direct_lstsq(a, b, cmplx=0), rtol=25 * _eps_cast(a1.dtype), atol=25 * _eps_cast(a1.dtype), err_msg='driver: %s' % lapack_driver)

    def test_random_complex_overdet(self):
        rng = np.random.RandomState(1234)
        for dtype in COMPLEX_DTYPES:
            for n, m in ((20, 15), (200, 2)):
                for lapack_driver in TestLstsq.lapack_drivers:
                    for overwrite in (True, False):
                        a = np.asarray(rng.random([n, m]) + 1j * rng.random([n, m]), dtype=dtype)
                        for i in range(m):
                            a[i, i] = 20 * (0.1 + a[i, i])
                        for i in range(2):
                            b = np.asarray(rng.random([n, 3]), dtype=dtype)
                            a1 = a.copy()
                            b1 = b.copy()
                            out = lstsq(a1, b1, lapack_driver=lapack_driver, overwrite_a=overwrite, overwrite_b=overwrite)
                            x = out[0]
                            r = out[2]
                            assert_(r == m, f'expected efficient rank {m}, got {r}')
                            assert_allclose(x, direct_lstsq(a, b, cmplx=1), rtol=25 * _eps_cast(a1.dtype), atol=25 * _eps_cast(a1.dtype), err_msg='driver: %s' % lapack_driver)

    def test_check_finite(self):
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "internal gelsd driver lwork query error,.*Falling back to 'gelss' driver.")
        at = np.array(((1, 20), (-30, 4)))
        for dtype, bt, lapack_driver, overwrite, check_finite in itertools.product(REAL_DTYPES, (((1, 0), (0, 1)), (1, 0), ((2, 1), (-30, 4))), TestLstsq.lapack_drivers, (True, False), (True, False)):
            a = at.astype(dtype)
            b = np.array(bt, dtype=dtype)
            a1 = a.copy()
            b1 = b.copy()
            out = lstsq(a1, b1, lapack_driver=lapack_driver, check_finite=check_finite, overwrite_a=overwrite, overwrite_b=overwrite)
            x = out[0]
            r = out[2]
            assert_(r == 2, 'expected efficient rank 2, got %s' % r)
            assert_allclose(dot(a, x), b, rtol=25 * _eps_cast(a.dtype), atol=25 * _eps_cast(a.dtype), err_msg='driver: %s' % lapack_driver)

    def test_zero_size(self):
        for a_shape, b_shape in (((0, 2), (0,)), ((0, 4), (0, 2)), ((4, 0), (4,)), ((4, 0), (4, 2))):
            b = np.ones(b_shape)
            x, residues, rank, s = lstsq(np.zeros(a_shape), b)
            assert_equal(x, np.zeros((a_shape[1],) + b_shape[1:]))
            residues_should_be = np.empty((0,)) if a_shape[1] else np.linalg.norm(b, axis=0) ** 2
            assert_equal(residues, residues_should_be)
            assert_(rank == 0, 'expected rank 0')
            assert_equal(s, np.empty((0,)))