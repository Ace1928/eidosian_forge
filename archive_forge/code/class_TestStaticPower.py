import copy
import itertools
import operator
import unittest
import numpy as np
from numba import jit, njit
from numba.core import types, utils, errors
from numba.core.types.functions import _header_lead
from numba.tests.support import TestCase, tag, needs_blas
from numba.tests.matmul_usecase import (matmul_usecase, imatmul_usecase,
class TestStaticPower(TestCase):
    """
    Test the ** operator with a static exponent, to exercise a
    dedicated optimization.
    """

    def _check_pow(self, exponents, values):
        for exp in exponents:
            regular_func = LiteralOperatorImpl.pow_usecase
            static_func = make_static_power(exp)
            static_cfunc = jit(nopython=True)(static_func)
            regular_cfunc = jit(nopython=True)(regular_func)
            for v in values:
                try:
                    expected = regular_cfunc(v, exp)
                except ZeroDivisionError:
                    with self.assertRaises(ZeroDivisionError):
                        static_cfunc(v)
                else:
                    got = static_cfunc(v)
                    self.assertPreciseEqual(expected, got, prec='double')

    def test_int_values(self):
        exponents = [1, 2, 3, 5, 17, 0, -1, -2, -3]
        vals = [0, 1, 3, -1, -4, np.int8(-3), np.uint16(4)]
        self._check_pow(exponents, vals)

    def test_real_values(self):
        exponents = [1, 2, 3, 5, 17, 0, -1, -2, -3, 1118481, -1118482]
        vals = [1.5, 3.25, -1.25, np.float32(-2.0), float('inf'), float('nan')]
        self._check_pow(exponents, vals)