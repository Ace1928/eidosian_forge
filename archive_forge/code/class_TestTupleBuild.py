import collections
import itertools
import numpy as np
from numba import njit, jit, typeof, literally
from numba.core import types, errors, utils
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
class TestTupleBuild(TestCase):

    def test_build_unpack(self):

        def check(p):
            pyfunc = lambda a: (1, *a)
            cfunc = jit(nopython=True)(pyfunc)
            self.assertPreciseEqual(cfunc(p), pyfunc(p))
        check((4, 5))
        check((4, 5.5))

    def test_build_unpack_assign_like(self):

        def check(p):
            pyfunc = lambda a: (*a,)
            cfunc = jit(nopython=True)(pyfunc)
            self.assertPreciseEqual(cfunc(p), pyfunc(p))
        check((4, 5))
        check((4, 5.5))

    def test_build_unpack_fail_on_list_assign_like(self):

        def check(p):
            pyfunc = lambda a: (*a,)
            cfunc = jit(nopython=True)(pyfunc)
            self.assertPreciseEqual(cfunc(p), pyfunc(p))
        with self.assertRaises(errors.TypingError) as raises:
            check([4, 5])
        msg1 = 'No implementation of function'
        self.assertIn(msg1, str(raises.exception))
        msg2 = 'tuple(reflected list('
        self.assertIn(msg2, str(raises.exception))

    def test_build_unpack_more(self):

        def check(p):
            pyfunc = lambda a: (1, *a, (1, 2), *a)
            cfunc = jit(nopython=True)(pyfunc)
            self.assertPreciseEqual(cfunc(p), pyfunc(p))
        check((4, 5))
        check((4, 5.5))

    def test_build_unpack_call(self):

        def check(p):

            @jit
            def inner(*args):
                return args
            pyfunc = lambda a: inner(1, *a)
            cfunc = jit(nopython=True)(pyfunc)
            self.assertPreciseEqual(cfunc(p), pyfunc(p))
        check((4, 5))
        check((4, 5.5))

    def test_build_unpack_call_more(self):

        def check(p):

            @jit
            def inner(*args):
                return args
            pyfunc = lambda a: inner(1, *a, *(1, 2), *a)
            cfunc = jit(nopython=True)(pyfunc)
            self.assertPreciseEqual(cfunc(p), pyfunc(p))
        check((4, 5))
        check((4, 5.5))

    def test_tuple_constructor(self):

        def check(pyfunc, arg):
            cfunc = jit(nopython=True)(pyfunc)
            self.assertPreciseEqual(cfunc(arg), pyfunc(arg))
        check(lambda _: tuple(), ())
        check(lambda a: tuple(a), (4, 5))
        check(lambda a: tuple(a), (4, 5.5))

    @unittest.skipIf(utils.PYVERSION < (3, 9), 'needs Python 3.9+')
    def test_unpack_with_predicate_fails(self):

        @njit
        def foo():
            a = (1,)
            b = (3, 2, 4)
            return (*(b if a[0] else (5, 6)),)
        with self.assertRaises(errors.UnsupportedError) as raises:
            foo()
        msg = 'op_LIST_EXTEND at the start of a block'
        self.assertIn(msg, str(raises.exception))

    def test_build_unpack_with_calls_in_unpack(self):

        def check(p):

            def pyfunc(a):
                z = [1, 2]
                return ((*a, z.append(3), z.extend(a), np.ones(3)), z)
            cfunc = jit(nopython=True)(pyfunc)
            self.assertPreciseEqual(cfunc(p), pyfunc(p))
        check((4, 5))

    def test_build_unpack_complicated(self):

        def check(p):

            def pyfunc(a):
                z = [1, 2]
                return ((*a, *(*a, a), *(a, (*(a, (1, 2), *(3,), *a), (a, 1, (2, 3), *a, 1), (1,))), *(z.append(4), z.extend(a))), z)
            cfunc = jit(nopython=True)(pyfunc)
            self.assertPreciseEqual(cfunc(p), pyfunc(p))
        check((10, 20))