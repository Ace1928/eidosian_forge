import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
class TestGetItem(TestCase):
    """
    Test basic indexed load from an array (returning a view or a scalar).
    Note fancy indexing is tested in test_fancy_indexing.
    """

    def test_1d_slicing(self, flags=enable_pyobj_flags):
        pyfunc = slicing_1d_usecase
        arraytype = types.Array(types.int32, 1, 'C')
        argtys = (arraytype, types.int32, types.int32, types.int32)
        cfunc = jit(argtys, **flags)(pyfunc)
        a = np.arange(10, dtype='i4')
        for indices in [(0, 10, 1), (2, 3, 1), (10, 0, 1), (0, 10, -1), (0, 10, 2), (9, 0, -1), (-5, -2, 1), (0, -1, 1)]:
            expected = pyfunc(a, *indices)
            self.assertPreciseEqual(cfunc(a, *indices), expected)

    def test_1d_slicing_npm(self):
        self.test_1d_slicing(flags=Noflags)

    def test_1d_slicing2(self, flags=enable_pyobj_flags):
        pyfunc = slicing_1d_usecase2
        arraytype = types.Array(types.int32, 1, 'C')
        argtys = (arraytype, types.int32, types.int32, types.int32)
        cfunc = jit(argtys, **flags)(pyfunc)
        a = np.arange(10, dtype='i4')
        args = [(0, 10, 1), (2, 3, 1), (10, 0, 1), (0, 10, -1), (0, 10, 2)]
        for arg in args:
            self.assertEqual(pyfunc(a, *arg), cfunc(a, *arg))
        arraytype = types.Array(types.int32, 1, 'A')
        argtys = (arraytype, types.int32, types.int32, types.int32)
        cfunc = jit(argtys, **flags)(pyfunc)
        a = np.arange(20, dtype='i4')[::2]
        self.assertFalse(a.flags['C_CONTIGUOUS'])
        self.assertFalse(a.flags['F_CONTIGUOUS'])
        args = [(0, 10, 1), (2, 3, 1), (10, 0, 1), (0, 10, -1), (0, 10, 2)]
        for arg in args:
            self.assertEqual(pyfunc(a, *arg), cfunc(a, *arg))

    def test_1d_slicing2_npm(self):
        self.test_1d_slicing2(flags=Noflags)

    def test_1d_slicing3(self, flags=enable_pyobj_flags):
        pyfunc = slicing_1d_usecase3
        arraytype = types.Array(types.int32, 1, 'C')
        argtys = (arraytype, types.int32, types.int32)
        cfunc = jit(argtys, **flags)(pyfunc)
        a = np.arange(10, dtype='i4')
        args = [(3, 10), (2, 3), (10, 0), (0, 10), (5, 10)]
        for arg in args:
            self.assertEqual(pyfunc(a, *arg), cfunc(a, *arg))
        arraytype = types.Array(types.int32, 1, 'A')
        argtys = (arraytype, types.int32, types.int32)
        cfunc = jit(argtys, **flags)(pyfunc)
        a = np.arange(20, dtype='i4')[::2]
        self.assertFalse(a.flags['C_CONTIGUOUS'])
        self.assertFalse(a.flags['F_CONTIGUOUS'])
        for arg in args:
            self.assertEqual(pyfunc(a, *arg), cfunc(a, *arg))

    def test_1d_slicing3_npm(self):
        self.test_1d_slicing3(flags=Noflags)

    def test_1d_slicing4(self, flags=enable_pyobj_flags):
        pyfunc = slicing_1d_usecase4
        arraytype = types.Array(types.int32, 1, 'C')
        argtys = (arraytype,)
        cfunc = jit(argtys, **flags)(pyfunc)
        a = np.arange(10, dtype='i4')
        self.assertEqual(pyfunc(a), cfunc(a))
        arraytype = types.Array(types.int32, 1, 'A')
        argtys = (arraytype,)
        cfunc = jit(argtys, **flags)(pyfunc)
        a = np.arange(20, dtype='i4')[::2]
        self.assertFalse(a.flags['C_CONTIGUOUS'])
        self.assertFalse(a.flags['F_CONTIGUOUS'])
        self.assertEqual(pyfunc(a), cfunc(a))

    def test_1d_slicing4_npm(self):
        self.test_1d_slicing4(flags=Noflags)

    def check_1d_slicing_with_arg(self, pyfunc, flags):
        args = list(range(-9, 10))
        arraytype = types.Array(types.int32, 1, 'C')
        argtys = (arraytype, types.int32)
        cfunc = jit(argtys, **flags)(pyfunc)
        a = np.arange(10, dtype='i4')
        for arg in args:
            self.assertEqual(pyfunc(a, arg), cfunc(a, arg))
        arraytype = types.Array(types.int32, 1, 'A')
        argtys = (arraytype, types.int32)
        cfunc = jit(argtys, **flags)(pyfunc)
        a = np.arange(20, dtype='i4')[::2]
        self.assertFalse(a.flags['C_CONTIGUOUS'])
        self.assertFalse(a.flags['F_CONTIGUOUS'])
        for arg in args:
            self.assertEqual(pyfunc(a, arg), cfunc(a, arg))

    def test_1d_slicing5(self, flags=enable_pyobj_flags):
        pyfunc = slicing_1d_usecase5
        self.check_1d_slicing_with_arg(pyfunc, flags)

    def test_1d_slicing5_npm(self):
        self.test_1d_slicing5(flags=Noflags)

    def test_1d_slicing6(self, flags=enable_pyobj_flags):
        pyfunc = slicing_1d_usecase6
        self.check_1d_slicing_with_arg(pyfunc, flags)

    def test_1d_slicing6_npm(self):
        self.test_1d_slicing6(flags=Noflags)

    def test_1d_slicing7(self, flags=enable_pyobj_flags):
        pyfunc = slicing_1d_usecase7
        self.check_1d_slicing_with_arg(pyfunc, flags)

    def test_1d_slicing7_npm(self):
        self.test_1d_slicing7(flags=Noflags)

    def test_1d_slicing8(self, flags=enable_pyobj_flags):
        pyfunc = slicing_1d_usecase8
        self.check_1d_slicing_with_arg(pyfunc, flags)

    def test_1d_slicing8_npm(self):
        self.test_1d_slicing8(flags=Noflags)

    def test_2d_slicing(self, flags=enable_pyobj_flags):
        """
        arr_2d[a:b:c]
        """
        pyfunc = slicing_1d_usecase
        arraytype = types.Array(types.int32, 2, 'C')
        argtys = (arraytype, types.int32, types.int32, types.int32)
        cfunc = jit(argtys, **flags)(pyfunc)
        a = np.arange(100, dtype='i4').reshape(10, 10)
        for args in [(0, 10, 1), (2, 3, 1), (10, 0, 1), (0, 10, -1), (0, 10, 2)]:
            self.assertPreciseEqual(pyfunc(a, *args), cfunc(a, *args), msg='for args %s' % (args,))

    def test_2d_slicing_npm(self):
        self.test_2d_slicing(flags=Noflags)

    def test_2d_slicing2(self, flags=enable_pyobj_flags):
        """
        arr_2d[a:b:c, d:e:f]
        """
        pyfunc = slicing_2d_usecase
        arraytype = types.Array(types.int32, 2, 'C')
        argtys = (arraytype, types.int32, types.int32, types.int32, types.int32, types.int32, types.int32)
        cfunc = jit(argtys, **flags)(pyfunc)
        a = np.arange(100, dtype='i4').reshape(10, 10)
        indices = [(0, 10, 1), (2, 3, 1), (10, 0, 1), (0, 10, -1), (0, 10, 2), (10, 0, -1), (9, 0, -2), (-5, -2, 1), (0, -1, 1)]
        args = [tup1 + tup2 for tup1, tup2 in itertools.product(indices, indices)]
        for arg in args:
            expected = pyfunc(a, *arg)
            self.assertPreciseEqual(cfunc(a, *arg), expected)
        arraytype = types.Array(types.int32, 2, 'A')
        argtys = (arraytype, types.int32, types.int32, types.int32, types.int32, types.int32, types.int32)
        cfunc = jit(argtys, **flags)(pyfunc)
        a = np.arange(400, dtype='i4').reshape(20, 20)[::2, ::2]
        for arg in args:
            expected = pyfunc(a, *arg)
            self.assertPreciseEqual(cfunc(a, *arg), expected)

    def test_2d_slicing2_npm(self):
        self.test_2d_slicing2(flags=Noflags)

    def test_2d_slicing3(self, flags=enable_pyobj_flags):
        """
        arr_2d[a:b:c, d]
        """
        pyfunc = slicing_2d_usecase3
        arraytype = types.Array(types.int32, 2, 'C')
        argtys = (arraytype, types.int32, types.int32, types.int32, types.int32)
        cfunc = jit(argtys, **flags)(pyfunc)
        a = np.arange(100, dtype='i4').reshape(10, 10)
        args = [(0, 10, 1, 0), (2, 3, 1, 1), (10, 0, -1, 8), (9, 0, -2, 4), (0, 10, 2, 3), (0, -1, 3, 1)]
        for arg in args:
            expected = pyfunc(a, *arg)
            self.assertPreciseEqual(cfunc(a, *arg), expected)
        arraytype = types.Array(types.int32, 2, 'A')
        argtys = (arraytype, types.int32, types.int32, types.int32, types.int32)
        cfunc = jit(argtys, **flags)(pyfunc)
        a = np.arange(400, dtype='i4').reshape(20, 20)[::2, ::2]
        for arg in args:
            expected = pyfunc(a, *arg)
            self.assertPreciseEqual(cfunc(a, *arg), expected)

    def test_2d_slicing3_npm(self):
        self.test_2d_slicing3(flags=Noflags)

    def test_3d_slicing(self, flags=enable_pyobj_flags):
        pyfunc = slicing_3d_usecase
        arraytype = types.Array(types.int32, 3, 'C')
        argtys = (arraytype, types.int32, types.int32, types.int32)
        cfunc = jit(argtys, **flags)(pyfunc)
        a = np.arange(1000, dtype='i4').reshape(10, 10, 10)
        args = [(0, 9, 1), (2, 3, 1), (9, 0, 1), (0, 9, -1), (0, 9, 2)]
        for arg in args:
            self.assertEqual(pyfunc(a, *arg), cfunc(a, *arg))
        arraytype = types.Array(types.int32, 3, 'A')
        argtys = (arraytype, types.int32, types.int32, types.int32)
        cfunc = jit(argtys, **flags)(pyfunc)
        a = np.arange(2000, dtype='i4')[::2].reshape(10, 10, 10)
        for arg in args:
            self.assertEqual(pyfunc(a, *arg), cfunc(a, *arg))

    def test_3d_slicing_npm(self):
        self.test_3d_slicing(flags=Noflags)

    def test_3d_slicing2(self, flags=enable_pyobj_flags):
        pyfunc = slicing_3d_usecase2
        arraytype = types.Array(types.int32, 3, 'C')
        argtys = (arraytype, types.int32, types.int32, types.int32)
        cfunc = jit(argtys, **flags)(pyfunc)
        a = np.arange(1000, dtype='i4').reshape(10, 10, 10)
        args = [(0, 9, 1), (2, 3, 1), (9, 0, 1), (0, 9, -1), (0, 9, 2)]
        for arg in args:
            self.assertEqual(pyfunc(a, *arg), cfunc(a, *arg))
        arraytype = types.Array(types.int32, 3, 'A')
        argtys = (arraytype, types.int32, types.int32, types.int32)
        cfunc = jit(argtys, **flags)(pyfunc)
        a = np.arange(2000, dtype='i4')[::2].reshape(10, 10, 10)
        for arg in args:
            self.assertEqual(pyfunc(a, *arg), cfunc(a, *arg))

    def test_3d_slicing2_npm(self):
        self.test_3d_slicing2(flags=Noflags)

    def test_1d_integer_indexing(self, flags=enable_pyobj_flags):
        pyfunc = integer_indexing_1d_usecase
        arraytype = types.Array(types.int32, 1, 'C')
        argtys = (arraytype, types.int32)
        cfunc = jit(argtys, **flags)(pyfunc)
        a = np.arange(10, dtype='i4')
        self.assertEqual(pyfunc(a, 0), cfunc(a, 0))
        self.assertEqual(pyfunc(a, 9), cfunc(a, 9))
        self.assertEqual(pyfunc(a, -1), cfunc(a, -1))
        arraytype = types.Array(types.int32, 1, 'A')
        argtys = (arraytype, types.int32)
        cfunc = jit(argtys, **flags)(pyfunc)
        a = np.arange(10, dtype='i4')[::2]
        self.assertFalse(a.flags['C_CONTIGUOUS'])
        self.assertFalse(a.flags['F_CONTIGUOUS'])
        self.assertEqual(pyfunc(a, 0), cfunc(a, 0))
        self.assertEqual(pyfunc(a, 2), cfunc(a, 2))
        self.assertEqual(pyfunc(a, -1), cfunc(a, -1))
        arraytype = types.Array(types.int32, 1, 'C')
        indextype = types.Array(types.int16, 0, 'C')
        argtys = (arraytype, indextype)
        cfunc = jit(argtys, **flags)(pyfunc)
        a = np.arange(3, 13, dtype=np.int32)
        for i in (0, 9, -2):
            idx = np.array(i).astype(np.int16)
            assert idx.ndim == 0
            self.assertEqual(pyfunc(a, idx), cfunc(a, idx))

    def test_1d_integer_indexing_npm(self):
        self.test_1d_integer_indexing(flags=Noflags)

    def test_integer_indexing_1d_for_2d(self, flags=enable_pyobj_flags):
        pyfunc = integer_indexing_1d_usecase
        arraytype = types.Array(types.int32, 2, 'C')
        argtys = (arraytype, types.int32)
        cfunc = jit(argtys, **flags)(pyfunc)
        a = np.arange(100, dtype='i4').reshape(10, 10)
        self.assertPreciseEqual(pyfunc(a, 0), cfunc(a, 0))
        self.assertPreciseEqual(pyfunc(a, 9), cfunc(a, 9))
        self.assertPreciseEqual(pyfunc(a, -1), cfunc(a, -1))
        arraytype = types.Array(types.int32, 2, 'A')
        argtys = (arraytype, types.int32)
        cfunc = jit(argtys, **flags)(pyfunc)
        a = np.arange(20, dtype='i4').reshape(5, 4)[::2]
        self.assertPreciseEqual(pyfunc(a, 0), cfunc(a, 0))

    def test_integer_indexing_1d_for_2d_npm(self):
        self.test_integer_indexing_1d_for_2d(flags=Noflags)

    def test_2d_integer_indexing(self, flags=enable_pyobj_flags, pyfunc=integer_indexing_2d_usecase):
        a = np.arange(100, dtype='i4').reshape(10, 10)
        arraytype = types.Array(types.int32, 2, 'C')
        argtys = (arraytype, types.int32, types.int32)
        cfunc = jit(argtys, **flags)(pyfunc)
        self.assertEqual(pyfunc(a, 0, 3), cfunc(a, 0, 3))
        self.assertEqual(pyfunc(a, 9, 9), cfunc(a, 9, 9))
        self.assertEqual(pyfunc(a, -2, -1), cfunc(a, -2, -1))
        a = np.arange(100, dtype='i4').reshape(10, 10)[::2, ::2]
        self.assertFalse(a.flags['C_CONTIGUOUS'])
        self.assertFalse(a.flags['F_CONTIGUOUS'])
        arraytype = types.Array(types.int32, 2, 'A')
        argtys = (arraytype, types.int32, types.int32)
        cfunc = jit(argtys, **flags)(pyfunc)
        self.assertEqual(pyfunc(a, 0, 1), cfunc(a, 0, 1))
        self.assertEqual(pyfunc(a, 2, 2), cfunc(a, 2, 2))
        self.assertEqual(pyfunc(a, -2, -1), cfunc(a, -2, -1))
        a = np.arange(100, dtype='i4').reshape(10, 10)
        arraytype = types.Array(types.int32, 2, 'C')
        indextype = types.Array(types.int32, 0, 'C')
        argtys = (arraytype, indextype, indextype)
        cfunc = jit(argtys, **flags)(pyfunc)
        for i, j in [(0, 3), (8, 9), (-2, -1)]:
            i = np.array(i).astype(np.int32)
            j = np.array(j).astype(np.int32)
            self.assertEqual(pyfunc(a, i, j), cfunc(a, i, j))

    def test_2d_integer_indexing_npm(self):
        self.test_2d_integer_indexing(flags=Noflags)

    def test_2d_integer_indexing2(self):
        self.test_2d_integer_indexing(pyfunc=integer_indexing_2d_usecase2)
        self.test_2d_integer_indexing(flags=Noflags, pyfunc=integer_indexing_2d_usecase2)

    def test_2d_integer_indexing_via_call(self):

        @njit
        def index1(X, i0):
            return X[i0]

        @njit
        def index2(X, i0, i1):
            return index1(X[i0], i1)
        a = np.arange(10).reshape(2, 5)
        self.assertEqual(index2(a, 0, 0), a[0][0])
        self.assertEqual(index2(a, 1, 1), a[1][1])
        self.assertEqual(index2(a, -1, -1), a[-1][-1])

    def test_2d_float_indexing(self, flags=enable_pyobj_flags):
        a = np.arange(100, dtype='i4').reshape(10, 10)
        pyfunc = integer_indexing_2d_usecase
        arraytype = types.Array(types.int32, 2, 'C')
        argtys = (arraytype, types.float32, types.int32)
        cfunc = jit(argtys, **flags)(pyfunc)
        self.assertEqual(pyfunc(a, 0, 0), cfunc(a, 0, 0))
        self.assertEqual(pyfunc(a, 9, 9), cfunc(a, 9, 9))
        self.assertEqual(pyfunc(a, -1, -1), cfunc(a, -1, -1))

    def test_partial_1d_indexing(self, flags=enable_pyobj_flags):
        pyfunc = partial_1d_usecase

        def check(arr, arraytype):
            argtys = (arraytype, types.int32)
            cfunc = jit(argtys, **flags)(pyfunc)
            self.assertEqual(pyfunc(arr, 0), cfunc(arr, 0))
            n = arr.shape[0] - 1
            self.assertEqual(pyfunc(arr, n), cfunc(arr, n))
            self.assertEqual(pyfunc(arr, -1), cfunc(arr, -1))
        a = np.arange(12, dtype='i4').reshape((4, 3))
        arraytype = types.Array(types.int32, 2, 'C')
        check(a, arraytype)
        a = np.arange(12, dtype='i4').reshape((3, 4)).T
        arraytype = types.Array(types.int32, 2, 'F')
        check(a, arraytype)
        a = np.arange(12, dtype='i4').reshape((3, 4))[::2]
        arraytype = types.Array(types.int32, 2, 'A')
        check(a, arraytype)

    def check_ellipsis(self, pyfunc, flags):

        def compile_func(arr):
            argtys = (typeof(arr), types.intp, types.intp)
            return jit(argtys, **flags)(pyfunc)

        def run(a):
            bounds = (0, 1, 2, -1, -2)
            cfunc = compile_func(a)
            for i, j in itertools.product(bounds, bounds):
                x = cfunc(a, i, j)
                np.testing.assert_equal(pyfunc(a, i, j), cfunc(a, i, j))
        run(np.arange(16, dtype='i4').reshape(4, 4))
        run(np.arange(27, dtype='i4').reshape(3, 3, 3))

    def test_ellipsis1(self, flags=enable_pyobj_flags):
        self.check_ellipsis(ellipsis_usecase1, flags)

    def test_ellipsis1_npm(self):
        self.test_ellipsis1(flags=Noflags)

    def test_ellipsis2(self, flags=enable_pyobj_flags):
        self.check_ellipsis(ellipsis_usecase2, flags)

    def test_ellipsis2_npm(self):
        self.test_ellipsis2(flags=Noflags)

    def test_ellipsis3(self, flags=enable_pyobj_flags):
        self.check_ellipsis(ellipsis_usecase3, flags)

    def test_ellipsis3_npm(self):
        self.test_ellipsis3(flags=Noflags)

    def test_ellipsis_issue1498(self):

        @njit
        def udt(arr):
            out = np.zeros_like(arr)
            i = 0
            for index, val in np.ndenumerate(arr[..., i]):
                out[index][i] = val
            return out
        py_func = udt.py_func
        outersize = 4
        innersize = 4
        arr = np.arange(outersize * innersize).reshape(outersize, innersize)
        got = udt(arr)
        expected = py_func(arr)
        np.testing.assert_equal(got, expected)

    def test_ellipsis_issue1499(self):

        @njit
        def udt(arr):
            return arr[..., 0]
        arr = np.arange(3)
        got = udt(arr)
        expected = udt.py_func(arr)
        np.testing.assert_equal(got, expected)

    def test_none_index(self, flags=enable_pyobj_flags):
        pyfunc = none_index_usecase
        arraytype = types.Array(types.int32, 2, 'C')
        argtys = (arraytype,)
        cfunc = jit(argtys, **flags)(pyfunc)
        a = np.arange(100, dtype='i4').reshape(10, 10)
        self.assertPreciseEqual(pyfunc(a), cfunc(a))

    def test_none_index_npm(self):
        with self.assertTypingError():
            self.test_none_index(flags=Noflags)

    def test_empty_tuple_indexing(self, flags=enable_pyobj_flags):
        pyfunc = empty_tuple_usecase
        arraytype = types.Array(types.int32, 0, 'C')
        argtys = (arraytype,)
        cfunc = jit(argtys, **flags)(pyfunc)
        a = np.arange(1, dtype='i4').reshape(())
        self.assertPreciseEqual(pyfunc(a), cfunc(a))

    def test_empty_tuple_indexing_npm(self):
        self.test_empty_tuple_indexing(flags=Noflags)