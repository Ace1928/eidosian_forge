import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
class TestSubtyping(TestCase):

    def setUp(self):
        self.value = 2
        a_dtype = np.dtype([('a', 'f8')])
        ab_dtype = np.dtype([('a', 'f8'), ('b', 'f8')])
        self.a_rec1 = np.array([1], dtype=a_dtype)[0]
        self.a_rec2 = np.array([2], dtype=a_dtype)[0]
        self.ab_rec1 = np.array([(self.value, 3)], dtype=ab_dtype)[0]
        self.ab_rec2 = np.array([(self.value + 1, 3)], dtype=ab_dtype)[0]
        self.func = lambda rec: rec['a']

    def test_common_field(self):
        njit_sig = njit(types.float64(typeof(self.a_rec1)))
        functions = [njit(self.func), njit_sig(self.func)]
        for fc in functions:
            fc(self.a_rec1)
            fc.disable_compile()
            y = fc(self.ab_rec1)
            self.assertEqual(self.value, y)

    def test_tuple_of_records(self):

        @njit
        def foo(rec_tup):
            x = 0
            for i in range(len(rec_tup)):
                x += rec_tup[i]['a']
            return x
        foo((self.a_rec1, self.a_rec2))
        foo.disable_compile()
        y = foo((self.ab_rec1, self.ab_rec2))
        self.assertEqual(2 * self.value + 1, y)

    def test_array_field(self):
        rec1 = np.empty(1, dtype=[('a', 'f8', (4,))])[0]
        rec1['a'][0] = 1
        rec2 = np.empty(1, dtype=[('a', 'f8', (4,)), ('b', 'f8')])[0]
        rec2['a'][0] = self.value

        @njit
        def foo(rec):
            return rec['a'][0]
        foo(rec1)
        foo.disable_compile()
        y = foo(rec2)
        self.assertEqual(self.value, y)

    def test_no_subtyping1(self):
        c_dtype = np.dtype([('c', 'f8')])
        c_rec1 = np.array([1], dtype=c_dtype)[0]

        @njit
        def foo(rec):
            return rec['c']
        foo(c_rec1)
        foo.disable_compile()
        with self.assertRaises(TypeError) as err:
            foo(self.a_rec1)
            self.assertIn('No matching definition for argument type(s) Record', str(err.exception))

    def test_no_subtyping2(self):
        jit_fc = njit(self.func)
        jit_fc(self.ab_rec1)
        jit_fc.disable_compile()
        with self.assertRaises(TypeError) as err:
            jit_fc(self.a_rec1)
            self.assertIn('No matching definition for argument type(s) Record', str(err.exception))

    def test_no_subtyping3(self):
        other_a_rec = np.array(['a'], dtype=np.dtype([('a', 'U25')]))[0]
        jit_fc = njit(self.func)
        jit_fc(self.a_rec1)
        jit_fc.disable_compile()
        with self.assertRaises(TypeError) as err:
            jit_fc(other_a_rec)
            self.assertIn('No matching definition for argument type(s) Record', str(err.exception))

    def test_branch_pruning(self):

        @njit
        def foo(rec, flag=None):
            n = 0
            n += rec['a']
            if flag is not None:
                n += rec['b']
                rec['b'] += 20
            return n
        self.assertEqual(foo(self.a_rec1), self.a_rec1[0])
        k = self.ab_rec1[1]
        self.assertEqual(foo(self.ab_rec1, flag=1), self.ab_rec1[0] + k)
        self.assertEqual(self.ab_rec1[1], k + 20)
        foo.disable_compile()
        self.assertEqual(len(foo.nopython_signatures), 2)
        self.assertEqual(foo(self.a_rec1) + 1, foo(self.ab_rec1))
        self.assertEqual(foo(self.ab_rec1, flag=1), self.ab_rec1[0] + k + 20)