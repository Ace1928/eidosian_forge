import numpy as np
from numba.np.ufunc.ufuncbuilder import GUFuncBuilder
from numba import vectorize, guvectorize
from numba.np.ufunc import PyUFunc_One
from numba.np.ufunc.dufunc import DUFunc as UFuncBuilder
from numba.tests.support import tag, TestCase
from numba.core import config
import unittest
class TestUfuncBuilding(TestCase):

    def test_basic_ufunc(self):
        from numba.tests.npyufunc.ufuncbuilding_usecases import add
        ufb = UFuncBuilder(add)
        cres = ufb.add('int32(int32, int32)')
        self.assertFalse(cres.objectmode)
        cres = ufb.add('int64(int64, int64)')
        self.assertFalse(cres.objectmode)
        ufunc = ufb.build_ufunc()

        def check(a):
            b = ufunc(a, a)
            self.assertPreciseEqual(a + a, b)
            self.assertEqual(b.dtype, a.dtype)
        a = np.arange(12, dtype='int32')
        check(a)
        a = a[::2]
        check(a)
        a = a.reshape((2, 3))
        check(a)
        self.assertEqual(ufunc.__name__, 'add')
        self.assertIn('An addition', ufunc.__doc__)

    def test_ufunc_struct(self):
        from numba.tests.npyufunc.ufuncbuilding_usecases import add
        ufb = UFuncBuilder(add)
        cres = ufb.add('complex64(complex64, complex64)')
        self.assertFalse(cres.objectmode)
        ufunc = ufb.build_ufunc()

        def check(a):
            b = ufunc(a, a)
            self.assertPreciseEqual(a + a, b)
            self.assertEqual(b.dtype, a.dtype)
        a = np.arange(12, dtype='complex64') + 1j
        check(a)
        a = a[::2]
        check(a)
        a = a.reshape((2, 3))
        check(a)

    def test_ufunc_forceobj(self):
        from numba.tests.npyufunc.ufuncbuilding_usecases import add
        ufb = UFuncBuilder(add, targetoptions={'forceobj': True})
        cres = ufb.add('int32(int32, int32)')
        self.assertTrue(cres.objectmode)
        ufunc = ufb.build_ufunc()
        a = np.arange(10, dtype='int32')
        b = ufunc(a, a)
        self.assertPreciseEqual(a + a, b)

    def test_nested_call(self):
        """
        Check nested call to an implicitly-typed ufunc.
        """
        from numba.tests.npyufunc.ufuncbuilding_usecases import outer
        builder = UFuncBuilder(outer, targetoptions={'nopython': True})
        builder.add('(int64, int64)')
        ufunc = builder.build_ufunc()
        self.assertEqual(ufunc(-1, 3), 2)

    def test_nested_call_explicit(self):
        """
        Check nested call to an explicitly-typed ufunc.
        """
        from numba.tests.npyufunc.ufuncbuilding_usecases import outer_explicit
        builder = UFuncBuilder(outer_explicit, targetoptions={'nopython': True})
        builder.add('(int64, int64)')
        ufunc = builder.build_ufunc()
        self.assertEqual(ufunc(-1, 3), 2)