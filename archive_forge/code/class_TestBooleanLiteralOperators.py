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
class TestBooleanLiteralOperators(TestCase):
    """
    Test operators with Boolean constants
    """

    def test_eq(self):

        def test_impl1(b):
            return a_val == b

        def test_impl2(a):
            return a == b_val

        def test_impl3():
            r1 = True == True
            r2 = True == False
            r3 = False == True
            r4 = False == False
            return (r1, r2, r3, r4)
        for a_val, b in itertools.product([True, False], repeat=2):
            cfunc1 = jit(nopython=True)(test_impl1)
            self.assertEqual(test_impl1(b), cfunc1(b))
        for a, b_val in itertools.product([True, False], repeat=2):
            cfunc2 = jit(nopython=True)(test_impl2)
            self.assertEqual(test_impl2(a), cfunc2(a))
        cfunc3 = jit(nopython=True)(test_impl3)
        self.assertEqual(test_impl3(), cfunc3())

    def test_ne(self):

        def test_impl1(b):
            return a_val != b

        def test_impl2(a):
            return a != b_val

        def test_impl3():
            r1 = True != True
            r2 = True != False
            r3 = False != True
            r4 = False != False
            return (r1, r2, r3, r4)
        for a_val, b in itertools.product([True, False], repeat=2):
            cfunc1 = jit(nopython=True)(test_impl1)
            self.assertEqual(test_impl1(b), cfunc1(b))
        for a, b_val in itertools.product([True, False], repeat=2):
            cfunc2 = jit(nopython=True)(test_impl2)
            self.assertEqual(test_impl2(a), cfunc2(a))
        cfunc3 = jit(nopython=True)(test_impl3)
        self.assertEqual(test_impl3(), cfunc3())

    def test_is(self):

        def test_impl1(b):
            return a_val is b

        def test_impl2():
            r1 = True is True
            r2 = True is False
            r3 = False is True
            r4 = False is False
            return (r1, r2, r3, r4)
        for a_val, b in itertools.product([True, False], repeat=2):
            cfunc1 = jit(nopython=True)(test_impl1)
            self.assertEqual(test_impl1(b), cfunc1(b))
        cfunc2 = jit(nopython=True)(test_impl2)
        self.assertEqual(test_impl2(), cfunc2())

    def test_not(self):

        def test_impl():
            a, b = (False, True)
            return (not a, not b)
        cfunc = jit(nopython=True)(test_impl)
        self.assertEqual(test_impl(), cfunc())

    def test_bool(self):

        def test_impl():
            a, b = (False, True)
            return (bool(a), bool(b))
        cfunc = jit(nopython=True)(test_impl)
        self.assertEqual(test_impl(), cfunc())

    def test_bool_to_str(self):

        def test_impl():
            a, b = (False, True)
            return (str(a), str(b))
        cfunc = jit(nopython=True)(test_impl)
        self.assertEqual(test_impl(), cfunc())