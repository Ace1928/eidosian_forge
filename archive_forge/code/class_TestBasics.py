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
class TestBasics(TestLinalgSystems):
    order1 = cycle(['F', 'C', 'C', 'F'])
    order2 = cycle(['C', 'F', 'C', 'F'])
    sizes = [(7, 1), (3, 3), (1, 7), (7,), (1,), (3,), 3.0, 5.0]

    def _assert_wrong_dim(self, rn, cfunc):
        self.assert_wrong_dimensions_1D(rn, cfunc, (np.array([[[1]]], dtype=np.float64), np.ones(1)), False)
        self.assert_wrong_dimensions_1D(rn, cfunc, (np.ones(1), np.array([[[1]]], dtype=np.float64)), False)

    def _gen_input(self, size, dtype, order):
        if not isinstance(size, tuple):
            return size
        elif len(size) == 1:
            return self.sample_vector(size[0], dtype)
        else:
            return self.sample_vector(size[0] * size[1], dtype).reshape(size, order=order)

    def _get_input(self, size1, size2, dtype):
        a = self._gen_input(size1, dtype, next(self.order1))
        b = self._gen_input(size2, dtype, next(self.order2))
        if np.iscomplexobj(a):
            b = b + 1j
        if np.iscomplexobj(b):
            a = a + 1j
        return (a, b)

    def test_outer(self):
        cfunc = jit(nopython=True)(outer_matrix)

        def check(a, b, **kwargs):
            expected = outer_matrix(a, b)
            got = cfunc(a, b)
            res = 5 * np.finfo(np.asarray(a).dtype).resolution
            np.testing.assert_allclose(got, expected, rtol=res, atol=res)
            if 'out' in kwargs:
                got = cfunc(a, b, **kwargs)
                np.testing.assert_allclose(got, expected, rtol=res, atol=res)
                np.testing.assert_allclose(kwargs['out'], expected, rtol=res, atol=res)
            with self.assertNoNRTLeak():
                cfunc(a, b, **kwargs)
        dts = cycle(self.dtypes)
        for size1, size2 in product(self.sizes, self.sizes):
            dtype = next(dts)
            a, b = self._get_input(size1, size2, dtype)
            check(a, b)
            c = np.empty((np.asarray(a).size, np.asarray(b).size), dtype=np.asarray(a).dtype)
            check(a, b, out=c)
        self._assert_wrong_dim('outer', cfunc)

    def test_kron(self):
        cfunc = jit(nopython=True)(kron_matrix)

        def check(a, b, **kwargs):
            expected = kron_matrix(a, b)
            got = cfunc(a, b)
            res = 5 * np.finfo(np.asarray(a).dtype).resolution
            np.testing.assert_allclose(got, expected, rtol=res, atol=res)
            with self.assertNoNRTLeak():
                cfunc(a, b)
        for size1, size2, dtype in product(self.sizes, self.sizes, self.dtypes):
            a, b = self._get_input(size1, size2, dtype)
            check(a, b)
        self._assert_wrong_dim('kron', cfunc)
        args = (np.empty(10)[::2], np.empty(10)[::2])
        msg = "only supports 'C' or 'F' layout"
        self.assert_error(cfunc, args, msg, err=errors.TypingError)