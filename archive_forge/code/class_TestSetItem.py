import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
class TestSetItem(TestCase):
    """
    Test basic indexed store into an array.
    Note fancy indexing is tested in test_fancy_indexing.
    """

    def test_conversion_setitem(self, flags=enable_pyobj_flags):
        """ this used to work, and was used in one of the tutorials """
        from numba import jit

        def pyfunc(array):
            for index in range(len(array)):
                array[index] = index % decimal.Decimal(100)
        cfunc = jit('void(i8[:])', **flags)(pyfunc)
        udt = np.arange(100, dtype='i1')
        control = udt.copy()
        pyfunc(control)
        cfunc(udt)
        self.assertPreciseEqual(udt, control)

    def test_1d_slicing_set(self, flags=enable_pyobj_flags):
        """
        1d to 1d slice assignment
        """
        pyfunc = slicing_1d_usecase_set
        dest_type = types.Array(types.int32, 1, 'C')
        src_type = types.Array(types.int16, 1, 'A')
        argtys = (dest_type, src_type, types.int32, types.int32, types.int32)
        cfunc = jit(argtys, **flags)(pyfunc)
        N = 10
        arg = np.arange(N, dtype='i2') + 40
        bounds = [0, 2, N - 2, N, N + 1, N + 3, -2, -N + 2, -N, -N - 1, -N - 3]

        def make_dest():
            return np.zeros_like(arg, dtype='i4')
        for start, stop in itertools.product(bounds, bounds):
            for step in (1, 2, -1, -2):
                args = (start, stop, step)
                index = slice(*args)
                pyleft = pyfunc(make_dest(), arg[index], *args)
                cleft = cfunc(make_dest(), arg[index], *args)
                self.assertPreciseEqual(pyleft, cleft)
        with self.assertRaises(ValueError):
            cfunc(np.zeros_like(arg, dtype=np.int32), arg, 0, 0, 1)

    def check_1d_slicing_set_sequence(self, flags, seqty, seq):
        """
        Generic sequence to 1d slice assignment
        """
        pyfunc = slicing_1d_usecase_set
        dest_type = types.Array(types.int32, 1, 'C')
        argtys = (dest_type, seqty, types.int32, types.int32, types.int32)
        cfunc = jit(argtys, **flags)(pyfunc).overloads[argtys].entry_point
        N = 10
        k = len(seq)
        arg = np.arange(N, dtype=np.int32)
        args = (seq, 1, -N + k + 1, 1)
        expected = pyfunc(arg.copy(), *args)
        got = cfunc(arg.copy(), *args)
        self.assertPreciseEqual(expected, got)
        args = (seq, 1, -N + k, 1)
        with self.assertRaises(ValueError) as raises:
            cfunc(arg.copy(), *args)

    def test_1d_slicing_set_tuple(self, flags=enable_pyobj_flags):
        """
        Tuple to 1d slice assignment
        """
        self.check_1d_slicing_set_sequence(flags, types.UniTuple(types.int16, 2), (8, -42))

    def test_1d_slicing_set_list(self, flags=enable_pyobj_flags):
        """
        List to 1d slice assignment
        """
        self.check_1d_slicing_set_sequence(flags, types.List(types.int16), [8, -42])

    def test_1d_slicing_broadcast(self, flags=enable_pyobj_flags):
        """
        scalar to 1d slice assignment
        """
        pyfunc = slicing_1d_usecase_set
        arraytype = types.Array(types.int32, 1, 'C')
        argtys = (arraytype, types.int16, types.int32, types.int32, types.int32)
        cfunc = jit(argtys, **flags)(pyfunc)
        N = 10
        arg = np.arange(N, dtype='i4')
        val = 42
        bounds = [0, 2, N - 2, N, N + 1, N + 3, -2, -N + 2, -N, -N - 1, -N - 3]
        for start, stop in itertools.product(bounds, bounds):
            for step in (1, 2, -1, -2):
                args = (val, start, stop, step)
                pyleft = pyfunc(arg.copy(), *args)
                cleft = cfunc(arg.copy(), *args)
                self.assertPreciseEqual(pyleft, cleft)

    def test_1d_slicing_add(self, flags=enable_pyobj_flags):
        pyfunc = slicing_1d_usecase_add
        arraytype = types.Array(types.int32, 1, 'C')
        argtys = (arraytype, arraytype, types.int32, types.int32)
        cfunc = jit(argtys, **flags)(pyfunc)
        arg = np.arange(10, dtype='i4')
        for test in ((0, 10), (2, 5)):
            pyleft = pyfunc(np.zeros_like(arg), arg[slice(*test)], *test)
            cleft = cfunc(np.zeros_like(arg), arg[slice(*test)], *test)
            self.assertPreciseEqual(pyleft, cleft)

    def test_1d_slicing_set_npm(self):
        self.test_1d_slicing_set(flags=Noflags)

    def test_1d_slicing_set_list_npm(self):
        self.test_1d_slicing_set_list(flags=Noflags)

    def test_1d_slicing_set_tuple_npm(self):
        self.test_1d_slicing_set_tuple(flags=Noflags)

    def test_1d_slicing_broadcast_npm(self):
        self.test_1d_slicing_broadcast(flags=Noflags)

    def test_1d_slicing_add_npm(self):
        self.test_1d_slicing_add(flags=Noflags)

    def test_2d_slicing_set(self, flags=enable_pyobj_flags):
        """
        2d to 2d slice assignment
        """
        pyfunc = slicing_2d_usecase_set
        arraytype = types.Array(types.int32, 2, 'A')
        argtys = (arraytype, arraytype, types.int32, types.int32, types.int32, types.int32, types.int32, types.int32)
        cfunc = jit(argtys, **flags)(pyfunc)
        arg = np.arange(10 * 10, dtype='i4').reshape(10, 10)
        tests = [(0, 10, 1, 0, 10, 1), (2, 3, 1, 2, 3, 1), (10, 0, 1, 10, 0, 1), (0, 10, -1, 0, 10, -1), (0, 10, 2, 0, 10, 2)]
        for test in tests:
            pyleft = pyfunc(np.zeros_like(arg), arg[slice(*test[0:3]), slice(*test[3:6])], *test)
            cleft = cfunc(np.zeros_like(arg), arg[slice(*test[0:3]), slice(*test[3:6])], *test)
            self.assertPreciseEqual(cleft, pyleft)

    def test_2d_slicing_broadcast(self, flags=enable_pyobj_flags):
        """
        scalar to 2d slice assignment
        """
        pyfunc = slicing_2d_usecase_set
        arraytype = types.Array(types.int32, 2, 'C')
        argtys = (arraytype, types.int16, types.int32, types.int32, types.int32, types.int32, types.int32, types.int32)
        cfunc = jit(argtys, **flags)(pyfunc)
        arg = np.arange(10 * 10, dtype='i4').reshape(10, 10)
        val = 42
        tests = [(0, 10, 1, 0, 10, 1), (2, 3, 1, 2, 3, 1), (10, 0, 1, 10, 0, 1), (0, 10, -1, 0, 10, -1), (0, 10, 2, 0, 10, 2)]
        for test in tests:
            pyleft = pyfunc(arg.copy(), val, *test)
            cleft = cfunc(arg.copy(), val, *test)
            self.assertPreciseEqual(cleft, pyleft)

    def test_2d_slicing_set_npm(self):
        self.test_2d_slicing_set(flags=Noflags)

    def test_2d_slicing_broadcast_npm(self):
        self.test_2d_slicing_broadcast(flags=Noflags)

    def test_setitem(self):
        """
        scalar indexed assignment
        """
        arr = np.arange(5)
        setitem_usecase(arr, 1, 42)
        self.assertEqual(arr.tolist(), [0, 42, 2, 3, 4])
        setitem_usecase(arr, np.array(3).astype(np.uint16), 8)
        self.assertEqual(arr.tolist(), [0, 42, 2, 8, 4])
        arr = np.arange(9).reshape(3, 3)
        setitem_usecase(arr, 1, 42)
        self.assertEqual(arr.tolist(), [[0, 1, 2], [42, 42, 42], [6, 7, 8]])

    def test_setitem_broadcast(self):
        """
        broadcasted array assignment
        """
        dst = np.arange(5)
        setitem_broadcast_usecase(dst, 42)
        self.assertEqual(dst.tolist(), [42] * 5)
        dst = np.arange(6).reshape(2, 3)
        setitem_broadcast_usecase(dst, np.arange(1, 4))
        self.assertEqual(dst.tolist(), [[1, 2, 3], [1, 2, 3]])
        dst = np.arange(6).reshape(2, 3)
        setitem_broadcast_usecase(dst, np.arange(1, 4).reshape(1, 3))
        self.assertEqual(dst.tolist(), [[1, 2, 3], [1, 2, 3]])
        dst = np.arange(12).reshape(2, 1, 2, 3)
        setitem_broadcast_usecase(dst, np.arange(1, 4).reshape(1, 3))
        inner2 = [[1, 2, 3], [1, 2, 3]]
        self.assertEqual(dst.tolist(), [[inner2]] * 2)
        dst = np.arange(5)
        setitem_broadcast_usecase(dst, np.arange(1, 6).reshape(1, 5))
        self.assertEqual(dst.tolist(), [1, 2, 3, 4, 5])
        dst = np.arange(6).reshape(2, 3)
        setitem_broadcast_usecase(dst, np.arange(1, 1 + dst.size).reshape(1, 1, 2, 3))
        self.assertEqual(dst.tolist(), [[1, 2, 3], [4, 5, 6]])

    def test_setitem_broadcast_error(self):
        dst = np.arange(5)
        src = np.arange(10).reshape(2, 5)
        with self.assertRaises(ValueError) as raises:
            setitem_broadcast_usecase(dst, src)
        errmsg = str(raises.exception)
        self.assertEqual('cannot broadcast source array for assignment', errmsg)
        dst = np.arange(5).reshape(1, 5)
        src = np.arange(10).reshape(1, 2, 5)
        with self.assertRaises(ValueError) as raises:
            setitem_broadcast_usecase(dst, src)
        errmsg = str(raises.exception)
        self.assertEqual('cannot assign slice from input of different size', errmsg)
        dst = np.arange(10).reshape(2, 5)
        src = np.arange(4)
        with self.assertRaises(ValueError) as raises:
            setitem_broadcast_usecase(dst, src)
        errmsg = str(raises.exception)
        self.assertEqual('cannot assign slice from input of different size', errmsg)

    def test_slicing_1d_broadcast(self):
        dst = np.arange(6).reshape(3, 2)
        src = np.arange(1, 3)
        slicing_1d_usecase_set(dst, src, 0, 2, 1)
        self.assertEqual(dst.tolist(), [[1, 2], [1, 2], [4, 5]])
        dst = np.arange(6).reshape(3, 2)
        src = np.arange(1, 3)
        slicing_1d_usecase_set(dst, src, 0, None, 2)
        self.assertEqual(dst.tolist(), [[1, 2], [2, 3], [1, 2]])
        dst = np.arange(6).reshape(3, 2)
        src = np.arange(1, 5).reshape(2, 2)
        slicing_1d_usecase_set(dst, src, None, 2, 1)
        self.assertEqual(dst.tolist(), [[1, 2], [3, 4], [4, 5]])

    def test_setitem_readonly(self):
        arr = np.arange(5)
        arr.flags.writeable = False
        with self.assertRaises((TypeError, errors.TypingError)) as raises:
            setitem_usecase(arr, 1, 42)
        self.assertIn('Cannot modify readonly array of type:', str(raises.exception))