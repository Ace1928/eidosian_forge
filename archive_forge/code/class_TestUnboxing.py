import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
class TestUnboxing(BaseTest):
    """
    Test unboxing of Python sets into native Numba sets.
    """

    @contextlib.contextmanager
    def assert_type_error(self, msg):
        with self.assertRaises(TypeError) as raises:
            yield
        if msg is not None:
            self.assertRegex(str(raises.exception), msg)

    def check_unary(self, pyfunc):
        cfunc = jit(nopython=True)(pyfunc)

        def check(arg):
            expected = pyfunc(arg)
            got = cfunc(arg)
            self.assertPreciseEqual(got, expected)
        return check

    def test_numbers(self):
        check = self.check_unary(unbox_usecase)
        check(set([1, 2]))
        check(set([1j, 2.5j]))
        check(set(range(100)))

    def test_tuples(self):
        check = self.check_unary(unbox_usecase2)
        check(set([(1, 2), (3, 4)]))
        check(set([(1, 2j), (3, 4j)]))

    def test_set_inside_tuple(self):
        check = self.check_unary(unbox_usecase3)
        check((1, set([2, 3, 4])))

    def test_set_of_tuples_inside_tuple(self):
        check = self.check_unary(unbox_usecase4)
        check((1, set([(2,), (3,)])))

    def test_errors(self):
        msg = "can't unbox heterogeneous set"
        pyfunc = noop
        cfunc = jit(nopython=True)(pyfunc)
        val = set([1, 2.5])
        with self.assert_type_error(msg):
            cfunc(val)
        self.assertEqual(val, set([1, 2.5]))
        with self.assert_type_error(msg):
            cfunc(set([1, 2j]))
        with self.assert_type_error(msg):
            cfunc((1, set([1, 2j])))
        with self.assert_type_error(msg):
            cfunc(Point(1, set([1, 2j])))
        lst = set([(1,), (2, 3)])
        with self.assertRaises((IndexError, ValueError)) as raises:
            cfunc(lst)