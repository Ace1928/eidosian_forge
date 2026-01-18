import unittest
import math
import sys
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase, tag
class IntWidthTest(TestCase):

    def check_nullary_func(self, pyfunc, **kwargs):
        cfunc = jit(**kwargs)(pyfunc)
        self.assertPreciseEqual(cfunc(), pyfunc())

    def test_global_uint64(self, nopython=False):
        pyfunc = usecase_uint64_global
        self.check_nullary_func(pyfunc, nopython=nopython)

    def test_global_uint64_npm(self):
        self.test_global_uint64(nopython=True)

    def test_constant_uint64(self, nopython=False):
        pyfunc = usecase_uint64_constant
        self.check_nullary_func(pyfunc, nopython=nopython)

    def test_constant_uint64_npm(self):
        self.test_constant_uint64(nopython=True)

    def test_constant_uint64_function_call(self, nopython=False):
        pyfunc = usecase_uint64_func
        self.check_nullary_func(pyfunc, nopython=nopython)

    def test_constant_uint64_function_call_npm(self):
        self.test_constant_uint64_function_call(nopython=True)

    def test_bit_length(self):
        f = utils.bit_length
        self.assertEqual(f(127), 7)
        self.assertEqual(f(-127), 7)
        self.assertEqual(f(128), 8)
        self.assertEqual(f(-128), 7)
        self.assertEqual(f(255), 8)
        self.assertEqual(f(-255), 8)
        self.assertEqual(f(256), 9)
        self.assertEqual(f(-256), 8)
        self.assertEqual(f(-257), 9)
        self.assertEqual(f(2147483647), 31)
        self.assertEqual(f(-2147483647), 31)
        self.assertEqual(f(-2147483648), 31)
        self.assertEqual(f(2147483648), 32)
        self.assertEqual(f(4294967295), 32)
        self.assertEqual(f(18446744073709551615), 64)
        self.assertEqual(f(18446744073709551616), 65)

    def test_constant_int64(self, nopython=False):
        self.check_nullary_func(usecase_int64_pos, nopython=nopython)
        self.check_nullary_func(usecase_int64_neg, nopython=nopython)
        self.check_nullary_func(usecase_int64_func, nopython=nopython)

    def test_constant_int64_npm(self):
        self.test_constant_int64(nopython=True)