from collections import namedtuple
import contextlib
import itertools
import math
import sys
import ctypes as ct
import numpy as np
from numba import jit, typeof, njit, literal_unroll, literally
import unittest
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.experimental import jitclass
from numba.core.extending import overload
class TestLists(MemoryLeakMixin, TestCase):

    def test_create_list(self):
        pyfunc = create_list
        cfunc = njit((types.int32, types.int32, types.int32))(pyfunc)
        self.assertEqual(cfunc(1, 2, 3), pyfunc(1, 2, 3))

    def test_create_nested_list(self):
        pyfunc = create_nested_list
        cfunc = njit((types.int32, types.int32, types.int32, types.int32, types.int32, types.int32))(pyfunc)
        self.assertEqual(cfunc(1, 2, 3, 4, 5, 6), pyfunc(1, 2, 3, 4, 5, 6))

    def check_unary_with_size(self, pyfunc, precise=True):
        cfunc = jit(nopython=True)(pyfunc)
        for n in [0, 3, 16, 70, 400]:
            eq = self.assertPreciseEqual if precise else self.assertEqual
            eq(cfunc(n), pyfunc(n))

    def test_constructor(self):
        self.check_unary_with_size(list_constructor)

    def test_constructor_empty(self):
        self.disable_leak_check()
        cfunc = jit(nopython=True)(list_constructor_empty)
        with self.assertRaises(errors.TypingError) as raises:
            cfunc()
        errmsg = str(raises.exception)
        self.assertIn('Cannot infer the type of variable', errmsg)
        self.assertIn('list(undefined)', errmsg)
        self.assertIn('For Numba to be able to compile a list', errmsg)

    def test_constructor_empty_but_typeable(self):
        args = [np.int32(1), 10.0, 1 + 3j, [7], [17.0, 14.0], np.array([10])]
        pyfunc = list_constructor_empty_but_typeable
        for arg in args:
            cfunc = jit(nopython=True)(pyfunc)
            expected = pyfunc(arg)
            got = cfunc(arg)
            self.assertPreciseEqual(got, expected)

    def test_append(self):
        self.check_unary_with_size(list_append)

    def test_append_heterogeneous(self):
        self.check_unary_with_size(list_append_heterogeneous, precise=False)

    def test_extend(self):
        self.check_unary_with_size(list_extend)

    def test_extend_heterogeneous(self):
        self.check_unary_with_size(list_extend_heterogeneous, precise=False)

    def test_pop0(self):
        self.check_unary_with_size(list_pop0)

    def test_pop1(self):
        pyfunc = list_pop1
        cfunc = jit(nopython=True)(pyfunc)
        for n in [5, 40]:
            for i in [0, 1, n - 2, n - 1, -1, -2, -n + 3, -n + 1]:
                expected = pyfunc(n, i)
                self.assertPreciseEqual(cfunc(n, i), expected)

    def test_pop_errors(self):
        self.disable_leak_check()
        cfunc = jit(nopython=True)(list_pop1)
        with self.assertRaises(IndexError) as cm:
            cfunc(0, 5)
        self.assertEqual(str(cm.exception), 'pop from empty list')
        with self.assertRaises(IndexError) as cm:
            cfunc(1, 5)
        self.assertEqual(str(cm.exception), 'pop index out of range')

    def test_insert(self):
        pyfunc = list_insert
        cfunc = jit(nopython=True)(pyfunc)
        for n in [5, 40]:
            indices = [0, 1, n - 2, n - 1, n + 1, -1, -2, -n + 3, -n - 1]
            for i in indices:
                expected = pyfunc(n, i, 42)
                self.assertPreciseEqual(cfunc(n, i, 42), expected)

    def test_len(self):
        self.check_unary_with_size(list_len)

    def test_getitem(self):
        self.check_unary_with_size(list_getitem)

    def test_setitem(self):
        self.check_unary_with_size(list_setitem)

    def check_slicing2(self, pyfunc):
        cfunc = jit(nopython=True)(pyfunc)
        sizes = [5, 40]
        for n in sizes:
            indices = [0, 1, n - 2, -1, -2, -n + 3, -n - 1, -n]
            for start, stop in itertools.product(indices, indices):
                expected = pyfunc(n, start, stop)
                self.assertPreciseEqual(cfunc(n, start, stop), expected)

    def test_getslice2(self):
        self.check_slicing2(list_getslice2)

    def test_setslice2(self):
        pyfunc = list_setslice2
        cfunc = jit(nopython=True)(pyfunc)
        sizes = [5, 40]
        for n, n_src in itertools.product(sizes, sizes):
            indices = [0, 1, n - 2, -1, -2, -n + 3, -n - 1, -n]
            for start, stop in itertools.product(indices, indices):
                expected = pyfunc(n, n_src, start, stop)
                self.assertPreciseEqual(cfunc(n, n_src, start, stop), expected)

    def test_getslice3(self):
        pyfunc = list_getslice3
        cfunc = jit(nopython=True)(pyfunc)
        for n in [10]:
            indices = [0, 1, n - 2, -1, -2, -n + 3, -n - 1, -n]
            steps = [4, 1, -1, 2, -3]
            for start, stop, step in itertools.product(indices, indices, steps):
                expected = pyfunc(n, start, stop, step)
                self.assertPreciseEqual(cfunc(n, start, stop, step), expected)

    def test_setslice3(self):
        pyfunc = list_setslice3
        cfunc = jit(nopython=True)(pyfunc)
        for n in [10]:
            indices = [0, 1, n - 2, -1, -2, -n + 3, -n - 1, -n]
            steps = [4, 1, -1, 2, -3]
            for start, stop, step in itertools.product(indices, indices, steps):
                expected = pyfunc(n, start, stop, step)
                self.assertPreciseEqual(cfunc(n, start, stop, step), expected)

    def test_setslice3_resize(self):
        self.disable_leak_check()
        pyfunc = list_setslice3_arbitrary
        cfunc = jit(nopython=True)(pyfunc)
        cfunc(5, 10, 0, 2, 1)
        with self.assertRaises(ValueError) as cm:
            cfunc(5, 100, 0, 3, 2)
        self.assertIn('cannot resize', str(cm.exception))

    def test_delslice0(self):
        self.check_unary_with_size(list_delslice0)

    def test_delslice1(self):
        self.check_slicing2(list_delslice1)

    def test_delslice2(self):
        self.check_slicing2(list_delslice2)

    def test_invalid_slice(self):
        self.disable_leak_check()
        pyfunc = list_getslice3
        cfunc = jit(nopython=True)(pyfunc)
        with self.assertRaises(ValueError) as cm:
            cfunc(10, 1, 2, 0)
        self.assertEqual(str(cm.exception), 'slice step cannot be zero')

    def test_iteration(self):
        self.check_unary_with_size(list_iteration)

    def test_reverse(self):
        self.check_unary_with_size(list_reverse)

    def test_contains(self):
        self.check_unary_with_size(list_contains)

    def check_index_result(self, pyfunc, cfunc, args):
        try:
            expected = pyfunc(*args)
        except ValueError:
            with self.assertRaises(ValueError):
                cfunc(*args)
        else:
            self.assertPreciseEqual(cfunc(*args), expected)

    def test_index1(self):
        self.disable_leak_check()
        pyfunc = list_index1
        cfunc = jit(nopython=True)(pyfunc)
        for v in (0, 1, 5, 10, 99999999):
            self.check_index_result(pyfunc, cfunc, (16, v))

    def test_index2(self):
        self.disable_leak_check()
        pyfunc = list_index2
        cfunc = jit(nopython=True)(pyfunc)
        n = 16
        for v in (0, 1, 5, 10, 99999999):
            indices = [0, 1, n - 2, n - 1, n + 1, -1, -2, -n + 3, -n - 1]
            for start in indices:
                self.check_index_result(pyfunc, cfunc, (16, v, start))

    def test_index3(self):
        self.disable_leak_check()
        pyfunc = list_index3
        cfunc = jit(nopython=True)(pyfunc)
        n = 16
        for v in (0, 1, 5, 10, 99999999):
            indices = [0, 1, n - 2, n - 1, n + 1, -1, -2, -n + 3, -n - 1]
            for start, stop in itertools.product(indices, indices):
                self.check_index_result(pyfunc, cfunc, (16, v, start, stop))

    def test_index_exception1(self):
        pyfunc = list_index3
        cfunc = jit(nopython=True)(pyfunc)
        msg = 'arg "start" must be an Integer.'
        with self.assertRaisesRegex(errors.TypingError, msg):
            cfunc(10, 0, 'invalid', 5)

    def test_index_exception2(self):
        pyfunc = list_index3
        cfunc = jit(nopython=True)(pyfunc)
        msg = 'arg "stop" must be an Integer.'
        with self.assertRaisesRegex(errors.TypingError, msg):
            cfunc(10, 0, 0, 'invalid')

    def test_remove(self):
        pyfunc = list_remove
        cfunc = jit(nopython=True)(pyfunc)
        n = 16
        for v in (0, 1, 5, 15):
            expected = pyfunc(n, v)
            self.assertPreciseEqual(cfunc(n, v), expected)

    def test_remove_error(self):
        self.disable_leak_check()
        pyfunc = list_remove
        cfunc = jit(nopython=True)(pyfunc)
        with self.assertRaises(ValueError) as cm:
            cfunc(10, 42)
        self.assertEqual(str(cm.exception), 'list.remove(x): x not in list')

    def test_count(self):
        pyfunc = list_count
        cfunc = jit(nopython=True)(pyfunc)
        for v in range(5):
            self.assertPreciseEqual(cfunc(18, v), pyfunc(18, v))

    def test_clear(self):
        self.check_unary_with_size(list_clear)

    def test_copy(self):
        self.check_unary_with_size(list_copy)

    def check_add(self, pyfunc):
        cfunc = jit(nopython=True)(pyfunc)
        sizes = [0, 3, 50, 300]
        for m, n in itertools.product(sizes, sizes):
            expected = pyfunc(m, n)
            self.assertPreciseEqual(cfunc(m, n), expected)

    def test_add(self):
        self.check_add(list_add)

    def test_add_heterogeneous(self):
        pyfunc = list_add_heterogeneous
        cfunc = jit(nopython=True)(pyfunc)
        expected = pyfunc()
        self.assertEqual(cfunc(), expected)

    def test_add_inplace(self):
        self.check_add(list_add_inplace)

    def test_add_inplace_heterogeneous(self):
        pyfunc = list_add_inplace_heterogeneous
        cfunc = jit(nopython=True)(pyfunc)
        expected = pyfunc()
        self.assertEqual(cfunc(), expected)

    def check_mul(self, pyfunc):
        cfunc = jit(nopython=True)(pyfunc)
        for n in [0, 3, 50, 300]:
            for v in [1, 2, 3, 0, -1, -42]:
                expected = pyfunc(n, v)
                self.assertPreciseEqual(cfunc(n, v), expected)

    def test_mul(self):
        self.check_mul(list_mul)

    def test_mul2(self):
        self.check_mul(list_mul2)

    def test_mul_inplace(self):
        self.check_mul(list_mul_inplace)

    @unittest.skipUnless(sys.maxsize >= 2 ** 32, 'need a 64-bit system to test for MemoryError')
    def test_mul_error(self):
        self.disable_leak_check()
        pyfunc = list_mul
        cfunc = jit(nopython=True)(pyfunc)
        with self.assertRaises(MemoryError):
            cfunc(1, 2 ** 58)
        if sys.platform.startswith('darwin'):
            libc = ct.CDLL('libc.dylib')
            libc.printf("###Please ignore the above error message i.e. can't allocate region. It is in fact the purpose of this test to request more memory than can be provided###\n".encode('UTF-8'))
        with self.assertRaises(MemoryError):
            cfunc(1, 2 ** 62)

    def test_bool(self):
        pyfunc = list_bool
        cfunc = jit(nopython=True)(pyfunc)
        for n in [0, 1, 3]:
            expected = pyfunc(n)
            self.assertPreciseEqual(cfunc(n), expected)

    def test_list_passing(self):

        @jit(nopython=True)
        def inner(lst):
            return (len(lst), lst[-1])

        @jit(nopython=True)
        def outer(n):
            l = list(range(n))
            return inner(l)
        self.assertPreciseEqual(outer(5), (5, 4))

    def _test_compare(self, pyfunc):

        def eq(args):
            self.assertIs(cfunc(*args), pyfunc(*args), 'mismatch for arguments %s' % (args,))
        cfunc = jit(nopython=True)(pyfunc)
        eq(((1, 2), (1, 2)))
        eq(((1, 2, 3), (1, 2)))
        eq(((1, 2), (1, 2, 3)))
        eq(((1, 2, 4), (1, 2, 3)))
        eq(((1.0, 2.0, 3.0), (1, 2, 3)))
        eq(((1.0, 2.0, 3.5), (1, 2, 3)))

    def test_eq(self):
        self._test_compare(eq_usecase)

    def test_ne(self):
        self._test_compare(ne_usecase)

    def test_le(self):
        self._test_compare(le_usecase)

    def test_lt(self):
        self._test_compare(lt_usecase)

    def test_ge(self):
        self._test_compare(ge_usecase)

    def test_gt(self):
        self._test_compare(gt_usecase)

    def test_identity(self):
        pyfunc = identity_usecase
        cfunc = jit(nopython=True)(pyfunc)
        self.assertPreciseEqual(cfunc(3), pyfunc(3))

    def test_bool_list(self):
        pyfunc = bool_list_usecase
        cfunc = jit(nopython=True)(pyfunc)
        self.assertPreciseEqual(cfunc(), pyfunc())