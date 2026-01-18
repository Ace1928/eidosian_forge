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
class TestLinalgMatrixPower(TestLinalgBase):
    """
    Tests for np.linalg.matrix_power.
    """

    def assert_int_exponenent(self, cfunc, args):
        cfunc(args[0], 1)
        with self.assertRaises(errors.TypingError):
            cfunc(*args)

    @needs_lapack
    def test_linalg_matrix_power(self):
        cfunc = jit(nopython=True)(matrix_power_matrix)

        def check(a, pwr):
            expected = matrix_power_matrix(a, pwr)
            got = cfunc(a, pwr)
            self.assert_contig_sanity(got, 'C')
            res = 5 * np.finfo(a.dtype).resolution
            np.testing.assert_allclose(got, expected, rtol=res, atol=res)
            with self.assertNoNRTLeak():
                cfunc(a, pwr)
        sizes = [(1, 1), (5, 5), (7, 7)]
        powers = [-33, -17] + list(range(-10, 10)) + [17, 33]
        for size, pwr, dtype, order in product(sizes, powers, self.dtypes, 'FC'):
            a = self.specific_sample_matrix(size, dtype, order)
            check(a, pwr)
            a = np.empty((0, 0), dtype=dtype, order=order)
            check(a, pwr)
        rn = 'matrix_power'
        self.assert_wrong_dtype(rn, cfunc, (np.ones((2, 2), dtype=np.int32), 1))
        self.assert_wrong_dtype(rn, cfunc, (np.ones((2, 2), dtype=np.int32), 1))
        args = (np.ones((3, 5)), 1)
        msg = 'input must be a square array'
        self.assert_error(cfunc, args, msg)
        self.assert_wrong_dimensions(rn, cfunc, (np.ones(10, dtype=np.float64), 1))
        self.assert_int_exponenent(cfunc, (np.ones((2, 2)), 1.2))
        self.assert_raise_on_singular(cfunc, (np.array([[0.0, 0], [1, 1]]), -1))