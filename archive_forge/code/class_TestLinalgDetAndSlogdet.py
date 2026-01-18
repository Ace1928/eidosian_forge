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
class TestLinalgDetAndSlogdet(TestLinalgBase):
    """
    Tests for np.linalg.det. and np.linalg.slogdet.
    Exactly the same inputs are used for both tests as
    det() is a trivial function of slogdet(), the tests
    are therefore combined.
    """

    def check_det(self, cfunc, a, **kwargs):
        expected = det_matrix(a, **kwargs)
        got = cfunc(a, **kwargs)
        resolution = 5 * np.finfo(a.dtype).resolution
        np.testing.assert_allclose(got, expected, rtol=resolution)
        with self.assertNoNRTLeak():
            cfunc(a, **kwargs)

    def check_slogdet(self, cfunc, a, **kwargs):
        expected = slogdet_matrix(a, **kwargs)
        got = cfunc(a, **kwargs)
        self.assertEqual(len(expected), len(got))
        self.assertEqual(len(got), 2)
        for k in range(2):
            self.assertEqual(np.iscomplexobj(got[k]), np.iscomplexobj(expected[k]))
        got_conv = a.dtype.type(got[0])
        np.testing.assert_array_almost_equal_nulp(got_conv, expected[0], nulp=10)
        resolution = 5 * np.finfo(a.dtype).resolution
        np.testing.assert_allclose(got[1], expected[1], rtol=resolution, atol=resolution)
        with self.assertNoNRTLeak():
            cfunc(a, **kwargs)

    def do_test(self, rn, check, cfunc):
        sizes = [(1, 1), (4, 4), (7, 7)]
        for size, dtype, order in product(sizes, self.dtypes, 'FC'):
            a = self.specific_sample_matrix(size, dtype, order)
            check(cfunc, a)
        for dtype, order in product(self.dtypes, 'FC'):
            a = np.zeros((3, 3), dtype=dtype)
            check(cfunc, a)
        check(cfunc, np.empty((0, 0)))
        self.assert_wrong_dtype(rn, cfunc, (np.ones((2, 2), dtype=np.int32),))
        self.assert_wrong_dimensions(rn, cfunc, (np.ones(10, dtype=np.float64),))
        self.assert_no_nan_or_inf(cfunc, (np.array([[1.0, 2.0], [np.inf, np.nan]], dtype=np.float64),))

    @needs_lapack
    def test_linalg_det(self):
        cfunc = jit(nopython=True)(det_matrix)
        self.do_test('det', self.check_det, cfunc)

    @needs_lapack
    def test_linalg_slogdet(self):
        cfunc = jit(nopython=True)(slogdet_matrix)
        self.do_test('slogdet', self.check_slogdet, cfunc)

    @needs_lapack
    def test_no_input_mutation(self):
        X = np.array([[1.0, 3, 2, 7], [-5, 4, 2, 3], [9, -3, 1, 1], [2, -2, 2, 8]], order='F')
        X_orig = np.copy(X)

        @jit(nopython=True)
        def func(X, test):
            if test:
                X = X[1:2, :]
            return np.linalg.slogdet(X)
        expected = func.py_func(X, False)
        np.testing.assert_allclose(X, X_orig)
        got = func(X, False)
        np.testing.assert_allclose(X, X_orig)
        np.testing.assert_allclose(expected, got)