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