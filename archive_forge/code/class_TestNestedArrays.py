import numpy as np
from numba import cuda
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support
class TestNestedArrays(CUDATestCase):

    def get_cfunc(self, pyfunc, retty):
        inner = cuda.jit(device=True)(pyfunc)

        @cuda.jit
        def outer(arg0, res):
            res[0] = inner(arg0)

        def host(arg0):
            res = np.zeros(1, dtype=retty)
            outer[1, 1](arg0, res)
            return res[0]
        return host

    def test_record_read_array(self):
        nbval = np.recarray(1, dtype=recordwitharray)
        nbval[0].h[0] = 15.0
        nbval[0].h[1] = 25.0
        cfunc = self.get_cfunc(record_read_array0, np.float32)
        res = cfunc(nbval[0])
        np.testing.assert_equal(res, nbval[0].h[0])
        cfunc = self.get_cfunc(record_read_array1, np.float32)
        res = cfunc(nbval[0])
        np.testing.assert_equal(res, nbval[0].h[1])

    def test_record_read_2d_array(self):
        nbval = np.recarray(1, dtype=recordwith2darray)
        nbval[0].j = np.asarray([1.5, 2.5, 3.5, 4.5, 5.5, 6.5], np.float32).reshape(3, 2)
        cfunc = self.get_cfunc(record_read_2d_array00, np.float32)
        res = cfunc(nbval[0])
        np.testing.assert_equal(res, nbval[0].j[0, 0])
        cfunc = self.get_cfunc(record_read_2d_array01, np.float32)
        res = cfunc(nbval[0])
        np.testing.assert_equal(res, nbval[0].j[0, 1])
        cfunc = self.get_cfunc(record_read_2d_array10, np.float32)
        res = cfunc(nbval[0])
        np.testing.assert_equal(res, nbval[0].j[1, 0])

    def test_setitem(self):

        def gen():
            nbarr1 = np.recarray(1, dtype=recordwith2darray)
            nbarr1[0] = np.array([(1, ((1, 2), (4, 5), (2, 3)))], dtype=recordwith2darray)[0]
            nbarr2 = np.recarray(1, dtype=recordwith2darray)
            nbarr2[0] = np.array([(10, ((10, 20), (40, 50), (20, 30)))], dtype=recordwith2darray)[0]
            return (nbarr1[0], nbarr2[0])
        pyfunc = record_setitem_array
        pyargs = gen()
        pyfunc(*pyargs)
        cfunc = cuda.jit(pyfunc)
        cuargs = gen()
        cfunc[1, 1](*cuargs)
        np.testing.assert_equal(pyargs, cuargs)

    def test_getitem_idx(self):
        nbarr = np.recarray(2, dtype=recordwitharray)
        nbarr[0] = np.array([(1, (2, 3))], dtype=recordwitharray)[0]
        for arg, retty in [(nbarr, recordwitharray), (nbarr[0], np.int32)]:
            pyfunc = recarray_getitem_return
            arr_expected = pyfunc(arg)
            cfunc = self.get_cfunc(pyfunc, retty)
            arr_res = cfunc(arg)
            np.testing.assert_equal(arr_res, arr_expected)

    @skip_on_cudasim('Structured array attr access not supported in simulator')
    def test_set_record(self):
        rec = np.ones(2, dtype=recordwith2darray).view(np.recarray)[0]
        nbarr = np.zeros(2, dtype=recordwith2darray).view(np.recarray)
        arr = np.zeros(2, dtype=recordwith2darray).view(np.recarray)
        pyfunc = recarray_set_record
        pyfunc(arr, rec)
        kernel = cuda.jit(pyfunc)
        kernel[1, 1](nbarr, rec)
        np.testing.assert_equal(nbarr, arr)

    def test_assign_array_to_nested(self):
        src = (np.arange(3) + 1).astype(np.int16)
        got = np.zeros(2, dtype=nested_array1_dtype)
        expected = np.zeros(2, dtype=nested_array1_dtype)
        pyfunc = assign_array_to_nested
        kernel = cuda.jit(pyfunc)
        kernel[1, 1](got[0], src)
        pyfunc(expected[0], src)
        np.testing.assert_array_equal(expected, got)

    def test_assign_array_to_nested_2d(self):
        src = (np.arange(6) + 1).astype(np.int16).reshape((3, 2))
        got = np.zeros(2, dtype=nested_array2_dtype)
        expected = np.zeros(2, dtype=nested_array2_dtype)
        pyfunc = assign_array_to_nested_2d
        kernel = cuda.jit(pyfunc)
        kernel[1, 1](got[0], src)
        pyfunc(expected[0], src)
        np.testing.assert_array_equal(expected, got)

    def test_issue_7693(self):
        src_dtype = np.dtype([('user', np.float64), ('array', np.int16, (3,))], align=True)
        dest_dtype = np.dtype([('user1', np.float64), ('array1', np.int16, (3,))], align=True)

        @cuda.jit
        def copy(index, src, dest):
            dest['user1'] = src[index]['user']
            dest['array1'] = src[index]['array']
        source = np.zeros(2, dtype=src_dtype)
        got = np.zeros(2, dtype=dest_dtype)
        expected = np.zeros(2, dtype=dest_dtype)
        source[0] = (1.2, [1, 2, 3])
        copy[1, 1](0, source, got[0])
        copy.py_func(0, source, expected[0])
        np.testing.assert_array_equal(expected, got)

    @unittest.expectedFailure
    def test_getitem_idx_2darray(self):
        nbarr = np.recarray(2, dtype=recordwith2darray)
        nbarr[0] = np.array([(1, ((1, 2), (4, 5), (2, 3)))], dtype=recordwith2darray)[0]
        for arg, retty in [(nbarr, recordwith2darray), (nbarr[0], (np.float32, (3, 2)))]:
            pyfunc = recarray_getitem_field_return2_2d
            arr_expected = pyfunc(arg)
            cfunc = self.get_cfunc(pyfunc, retty)
            arr_res = cfunc(arg)
            np.testing.assert_equal(arr_res, arr_expected)

    @unittest.expectedFailure
    def test_return_getattr_getitem_fieldname(self):
        nbarr = np.recarray(2, dtype=recordwitharray)
        nbarr[0] = np.array([(1, (2, 3))], dtype=recordwitharray)[0]
        for arg, retty in [(nbarr, recordwitharray), (nbarr[0], np.float32)]:
            for pyfunc in [recarray_getitem_field_return, recarray_getitem_field_return2]:
                arr_expected = pyfunc(arg)
                cfunc = self.get_cfunc(pyfunc, retty)
                arr_res = cfunc(arg)
                np.testing.assert_equal(arr_res, arr_expected)

    @unittest.expectedFailure
    def test_record_read_arrays(self):
        nbval = np.recarray(2, dtype=recordwitharray)
        nbval[0].h[0] = 15.0
        nbval[0].h[1] = 25.0
        nbval[1].h[0] = 35.0
        nbval[1].h[1] = 45.4
        cfunc = self.get_cfunc(record_read_whole_array, np.float32)
        res = cfunc(nbval)
        np.testing.assert_equal(res, nbval.h)

    @unittest.expectedFailure
    def test_return_array(self):
        nbval = np.recarray(2, dtype=recordwitharray)
        nbval[0] = np.array([(1, (2, 3))], dtype=recordwitharray)[0]
        pyfunc = record_read_array0
        arr_expected = pyfunc(nbval)
        cfunc = self.get_cfunc(pyfunc, np.float32)
        arr_res = cfunc(nbval)
        np.testing.assert_equal(arr_expected, arr_res)

    @skip_on_cudasim('Will unexpectedly pass on cudasim')
    @unittest.expectedFailure
    def test_set_array(self):
        arr = np.zeros(2, dtype=recordwith2darray).view(np.recarray)
        rec = arr[0]
        nbarr = np.zeros(2, dtype=recordwith2darray).view(np.recarray)
        nbrec = nbarr[0]
        for pyfunc in (record_write_full_array, record_write_full_array_alt):
            pyfunc(rec)
            kernel = cuda.jit(pyfunc)
            kernel[1, 1](nbrec)
            np.testing.assert_equal(nbarr, arr)

    @unittest.expectedFailure
    def test_set_arrays(self):
        arr = np.zeros(2, dtype=recordwith2darray).view(np.recarray)
        nbarr = np.zeros(2, dtype=recordwith2darray).view(np.recarray)
        for pyfunc in (recarray_write_array_of_nestedarray_broadcast, recarray_write_array_of_nestedarray):
            arr_expected = pyfunc(arr)
            cfunc = self.get_cfunc(pyfunc, nbarr.dtype)
            arr_res = cfunc(nbarr)
            np.testing.assert_equal(arr_res, arr_expected)