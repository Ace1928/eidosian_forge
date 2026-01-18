import contextlib
import gc
from itertools import product, cycle
import sys
import warnings
from numbers import Number, Integral
import platform
import numpy as np
from numba import jit, njit, typeof
from numba.core import errors
from numba.tests.support import (TestCase, tag, needs_lapack, needs_blas,
from .matmul_usecase import matmul_usecase
import unittest
class TestLinalgSvd(TestLinalgBase):
    """
    Tests for np.linalg.svd.
    """

    def check_reconstruction(self, a, got, expected):
        u, sv, vt = got
        for k in range(len(expected)):
            self.assertEqual(got[k].shape, expected[k].shape)
        s = np.zeros((u.shape[1], vt.shape[0]))
        np.fill_diagonal(s, sv)
        rec = np.dot(np.dot(u, s), vt)
        resolution = np.finfo(a.dtype).resolution
        np.testing.assert_allclose(a, rec, rtol=10 * resolution, atol=100 * resolution)

    @needs_lapack
    def test_linalg_svd(self):
        """
        Test np.linalg.svd
        """
        cfunc = jit(nopython=True)(svd_matrix)

        def check(a, **kwargs):
            expected = svd_matrix(a, **kwargs)
            got = cfunc(a, **kwargs)
            self.assertEqual(len(expected), len(got))
            self.assertEqual(len(got), 3)
            self.assert_contig_sanity(got, 'F')
            use_reconstruction = False
            for k in range(len(expected)):
                try:
                    np.testing.assert_array_almost_equal_nulp(got[k], expected[k], nulp=10)
                except AssertionError:
                    use_reconstruction = True
            if use_reconstruction:
                self.check_reconstruction(a, got, expected)
            with self.assertNoNRTLeak():
                cfunc(a, **kwargs)
        sizes = [(7, 1), (7, 5), (5, 7), (3, 3), (1, 7)]
        full_matrices = (True, False)
        for size, dtype, fmat, order in product(sizes, self.dtypes, full_matrices, 'FC'):
            a = self.specific_sample_matrix(size, dtype, order)
            check(a, full_matrices=fmat)
        rn = 'svd'
        self.assert_wrong_dtype(rn, cfunc, (np.ones((2, 2), dtype=np.int32),))
        self.assert_wrong_dimensions(rn, cfunc, (np.ones(10, dtype=np.float64),))
        self.assert_no_nan_or_inf(cfunc, (np.array([[1.0, 2.0], [np.inf, np.nan]], dtype=np.float64),))
        for sz in [(0, 1), (1, 0), (0, 0)]:
            args = (np.empty(sz), True)
            self.assert_raise_on_empty(cfunc, args)

    @needs_lapack
    def test_no_input_mutation(self):
        X = np.array([[1.0, 3, 2, 7], [-5, 4, 2, 3], [9, -3, 1, 1], [2, -2, 2, 8]], order='F')
        X_orig = np.copy(X)

        @jit(nopython=True)
        def func(X, test):
            if test:
                X = X[1:2, :]
            return np.linalg.svd(X)
        expected = func.py_func(X, False)
        np.testing.assert_allclose(X, X_orig)
        got = func(X, False)
        np.testing.assert_allclose(X, X_orig)
        try:
            for e_a, g_a in zip(expected, got):
                np.testing.assert_allclose(e_a, g_a)
        except AssertionError:
            self.check_reconstruction(X, got, expected)