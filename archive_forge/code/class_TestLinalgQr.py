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
class TestLinalgQr(TestLinalgBase):
    """
    Tests for np.linalg.qr.
    """

    @needs_lapack
    def test_linalg_qr(self):
        """
        Test np.linalg.qr
        """
        cfunc = jit(nopython=True)(qr_matrix)

        def check(a, **kwargs):
            expected = qr_matrix(a, **kwargs)
            got = cfunc(a, **kwargs)
            self.assertEqual(len(expected), len(got))
            self.assertEqual(len(got), 2)
            self.assert_contig_sanity(got, 'F')
            use_reconstruction = False
            for k in range(len(expected)):
                try:
                    np.testing.assert_array_almost_equal_nulp(got[k], expected[k], nulp=10)
                except AssertionError:
                    use_reconstruction = True
            if use_reconstruction:
                q, r = got
                for k in range(len(expected)):
                    self.assertEqual(got[k].shape, expected[k].shape)
                rec = np.dot(q, r)
                resolution = np.finfo(a.dtype).resolution
                np.testing.assert_allclose(a, rec, rtol=10 * resolution, atol=100 * resolution)
                self.assert_is_identity_matrix(np.dot(np.conjugate(q.T), q))
            with self.assertNoNRTLeak():
                cfunc(a, **kwargs)
        sizes = [(7, 1), (11, 5), (5, 11), (3, 3), (1, 7)]
        for size, dtype, order in product(sizes, self.dtypes, 'FC'):
            a = self.specific_sample_matrix(size, dtype, order)
            check(a)
        rn = 'qr'
        self.assert_wrong_dtype(rn, cfunc, (np.ones((2, 2), dtype=np.int32),))
        self.assert_wrong_dimensions(rn, cfunc, (np.ones(10, dtype=np.float64),))
        self.assert_no_nan_or_inf(cfunc, (np.array([[1.0, 2.0], [np.inf, np.nan]], dtype=np.float64),))
        for sz in [(0, 1), (1, 0), (0, 0)]:
            self.assert_raise_on_empty(cfunc, (np.empty(sz),))

    @needs_lapack
    def test_no_input_mutation(self):
        X = np.array([[1.0, 3, 2, 7], [-5, 4, 2, 3], [9, -3, 1, 1], [2, -2, 2, 8]], order='F')
        X_orig = np.copy(X)

        @jit(nopython=True)
        def func(X, test):
            if test:
                X = X[1:2, :]
            return np.linalg.qr(X)
        expected = func.py_func(X, False)
        np.testing.assert_allclose(X, X_orig)
        got = func(X, False)
        np.testing.assert_allclose(X, X_orig)
        for e_a, g_a in zip(expected, got):
            np.testing.assert_allclose(e_a, g_a)