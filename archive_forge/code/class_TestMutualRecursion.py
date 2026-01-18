import math
import warnings
from numba import jit
from numba.core.errors import TypingError, NumbaWarning
from numba.tests.support import TestCase
import unittest
class TestMutualRecursion(TestCase):

    def test_mutual_1(self):
        from numba.tests.recursion_usecases import outer_fac
        expect = math.factorial(10)
        self.assertPreciseEqual(outer_fac(10), expect)

    def test_mutual_2(self):
        from numba.tests.recursion_usecases import make_mutual2
        pfoo, pbar = make_mutual2()
        cfoo, cbar = make_mutual2(jit(nopython=True))
        for x in [-1, 0, 1, 3]:
            self.assertPreciseEqual(pfoo(x=x), cfoo(x=x))
            self.assertPreciseEqual(pbar(y=x, z=1), cbar(y=x, z=1))

    def test_runaway(self):
        from numba.tests.recursion_usecases import runaway_mutual
        with self.assertRaises(TypingError) as raises:
            runaway_mutual(123)
        self.assertIn('cannot type infer runaway recursion', str(raises.exception))

    def test_type_change(self):
        from numba.tests.recursion_usecases import make_type_change_mutual
        pfunc = make_type_change_mutual()
        cfunc = make_type_change_mutual(jit(nopython=True))
        args = (13, 0.125)
        self.assertPreciseEqual(pfunc(*args), cfunc(*args))

    def test_four_level(self):
        from numba.tests.recursion_usecases import make_four_level
        pfunc = make_four_level()
        cfunc = make_four_level(jit(nopython=True))
        arg = 7
        self.assertPreciseEqual(pfunc(arg), cfunc(arg))

    def test_inner_error(self):
        from numba.tests.recursion_usecases import make_inner_error
        cfunc = make_inner_error(jit(nopython=True))
        with self.assertRaises(TypingError) as raises:
            cfunc(2)
        errmsg = "Unknown attribute 'ndim'"
        self.assertIn(errmsg, str(raises.exception))
        cfunc = make_inner_error(jit(forceobj=True))
        pfunc = make_inner_error()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=NumbaWarning)
            got = cfunc(6)
        self.assertEqual(got, pfunc(6))

    def test_raise(self):
        from numba.tests.recursion_usecases import make_raise_mutual
        cfunc = make_raise_mutual()
        with self.assertRaises(ValueError) as raises:
            cfunc(2)
        self.assertEqual(str(raises.exception), 'raise_mutual')