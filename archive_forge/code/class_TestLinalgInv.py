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
class TestLinalgInv(TestLinalgBase):
    """
    Tests for np.linalg.inv.
    """

    @needs_lapack
    def test_linalg_inv(self):
        """
        Test np.linalg.inv
        """
        n = 10
        cfunc = jit(nopython=True)(invert_matrix)

        def check(a, **kwargs):
            expected = invert_matrix(a)
            got = cfunc(a)
            self.assert_contig_sanity(got, 'F')
            use_reconstruction = False
            try:
                np.testing.assert_array_almost_equal_nulp(got, expected, nulp=10)
            except AssertionError:
                use_reconstruction = True
            if use_reconstruction:
                rec = np.dot(got, a)
                self.assert_is_identity_matrix(rec)
            with self.assertNoNRTLeak():
                cfunc(a)
        for dtype, order in product(self.dtypes, 'CF'):
            a = self.specific_sample_matrix((n, n), dtype, order)
            check(a)
        check(np.empty((0, 0)))
        self.assert_non_square(cfunc, (np.ones((2, 3)),))
        self.assert_wrong_dtype('inv', cfunc, (np.ones((2, 2), dtype=np.int32),))
        self.assert_wrong_dimensions('inv', cfunc, (np.ones(10),))
        self.assert_raise_on_singular(cfunc, (np.zeros((2, 2)),))

    @needs_lapack
    def test_no_input_mutation(self):
        X = np.array([[1.0, 3, 2, 7], [-5, 4, 2, 3], [9, -3, 1, 1], [2, -2, 2, 8]], order='F')
        X_orig = np.copy(X)

        @jit(nopython=True)
        def ainv(X, test):
            if test:
                X = X[1:2, :]
            return np.linalg.inv(X)
        expected = ainv.py_func(X, False)
        np.testing.assert_allclose(X, X_orig)
        got = ainv(X, False)
        np.testing.assert_allclose(X, X_orig)
        np.testing.assert_allclose(expected, got)