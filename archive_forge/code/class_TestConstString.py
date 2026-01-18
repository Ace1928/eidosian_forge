import re
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, skip_on_cudasim, CUDATestCase
from llvmlite import ir
class TestConstString(CUDATestCase):

    def test_assign_const_unicode_string(self):

        @cuda.jit
        def str_assign(arr):
            i = cuda.grid(1)
            if i < len(arr):
                arr[i] = 'XYZ'
        n_strings = 8
        arr = np.zeros(n_strings + 1, dtype='<U12')
        str_assign[1, n_strings](arr)
        expected = np.zeros_like(arr)
        expected[:-1] = 'XYZ'
        expected[-1] = ''
        np.testing.assert_equal(arr, expected)

    def test_assign_const_byte_string(self):

        @cuda.jit
        def bytes_assign(arr):
            i = cuda.grid(1)
            if i < len(arr):
                arr[i] = b'XYZ'
        n_strings = 8
        arr = np.zeros(n_strings + 1, dtype='S12')
        bytes_assign[1, n_strings](arr)
        expected = np.zeros_like(arr)
        expected[:-1] = b'XYZ'
        expected[-1] = b''
        np.testing.assert_equal(arr, expected)

    def test_assign_const_string_in_record(self):

        @cuda.jit
        def f(a):
            a[0]['x'] = 1
            a[0]['y'] = 'ABC'
            a[1]['x'] = 2
            a[1]['y'] = 'XYZ'
        dt = np.dtype([('x', np.int32), ('y', np.dtype('<U12'))])
        a = np.zeros(2, dt)
        f[1, 1](a)
        reference = np.asarray([(1, 'ABC'), (2, 'XYZ')], dtype=dt)
        np.testing.assert_array_equal(reference, a)

    def test_assign_const_bytes_in_record(self):

        @cuda.jit
        def f(a):
            a[0]['x'] = 1
            a[0]['y'] = b'ABC'
            a[1]['x'] = 2
            a[1]['y'] = b'XYZ'
        dt = np.dtype([('x', np.float32), ('y', np.dtype('S12'))])
        a = np.zeros(2, dt)
        f[1, 1](a)
        reference = np.asarray([(1, b'ABC'), (2, b'XYZ')], dtype=dt)
        np.testing.assert_array_equal(reference, a)