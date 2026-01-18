from numba import njit
from functools import reduce
import unittest
class TestReduce(unittest.TestCase):

    def test_basic_reduce_external_func(self):
        func = njit(lambda x, y: x + y)

        def impl():
            return reduce(func, range(-10, 10))
        cfunc = njit(impl)
        self.assertEqual(impl(), cfunc())

    def test_basic_reduce_closure(self):

        def impl():

            def func(x, y):
                return x + y
            return reduce(func, range(-10, 10), 100)
        cfunc = njit(impl)
        self.assertEqual(impl(), cfunc())