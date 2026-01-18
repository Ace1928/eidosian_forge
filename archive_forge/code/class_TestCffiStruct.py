import ctypes
import os
import subprocess
import sys
from collections import namedtuple
import numpy as np
from numba import cfunc, carray, farray, njit
from numba.core import types, typing, utils
import numba.core.typing.cffi_utils as cffi_support
from numba.tests.support import (TestCase, skip_unless_cffi, tag,
import unittest
from numba.np import numpy_support
@skip_unless_cffi
class TestCffiStruct(TestCase):
    c_source = '\ntypedef struct _big_struct {\n    int    i1;\n    float  f2;\n    double d3;\n    float  af4[9];\n} big_struct;\n\ntypedef struct _error {\n    int bits:4;\n} error;\n\ntypedef double (*myfunc)(big_struct*, size_t);\n'

    def get_ffi(self, src=c_source):
        from cffi import FFI
        ffi = FFI()
        ffi.cdef(src)
        return ffi

    def test_type_parsing(self):
        ffi = self.get_ffi()
        big_struct = ffi.typeof('big_struct')
        nbtype = cffi_support.map_type(big_struct, use_record_dtype=True)
        self.assertIsInstance(nbtype, types.Record)
        self.assertEqual(len(nbtype), 4)
        self.assertEqual(nbtype.typeof('i1'), types.int32)
        self.assertEqual(nbtype.typeof('f2'), types.float32)
        self.assertEqual(nbtype.typeof('d3'), types.float64)
        self.assertEqual(nbtype.typeof('af4'), types.NestedArray(dtype=types.float32, shape=(9,)))
        myfunc = ffi.typeof('myfunc')
        sig = cffi_support.map_type(myfunc, use_record_dtype=True)
        self.assertIsInstance(sig, typing.Signature)
        self.assertEqual(sig.args[0], types.CPointer(nbtype))
        self.assertEqual(sig.args[1], types.uintp)
        self.assertEqual(sig.return_type, types.float64)

    def test_cfunc_callback(self):
        ffi = self.get_ffi()
        big_struct = ffi.typeof('big_struct')
        nb_big_struct = cffi_support.map_type(big_struct, use_record_dtype=True)
        sig = cffi_support.map_type(ffi.typeof('myfunc'), use_record_dtype=True)

        @njit
        def calc(base):
            tmp = 0
            for i in range(base.size):
                elem = base[i]
                tmp += elem.i1 * elem.f2 / elem.d3
                tmp += base[i].af4.sum()
            return tmp

        @cfunc(sig)
        def foo(ptr, n):
            base = carray(ptr, n)
            return calc(base)
        mydata = ffi.new('big_struct[3]')
        ptr = ffi.cast('big_struct*', mydata)
        for i in range(3):
            ptr[i].i1 = i * 123
            ptr[i].f2 = i * 213
            ptr[i].d3 = (1 + i) * 213
            for j in range(9):
                ptr[i].af4[j] = i * 10 + j
        addr = int(ffi.cast('size_t', ptr))
        got = foo.ctypes(addr, 3)
        array = np.ndarray(buffer=ffi.buffer(mydata), dtype=numpy_support.as_dtype(nb_big_struct), shape=3)
        expect = calc(array)
        self.assertEqual(got, expect)

    def test_unsupport_bitsize(self):
        ffi = self.get_ffi()
        with self.assertRaises(ValueError) as raises:
            cffi_support.map_type(ffi.typeof('error'), use_record_dtype=True)
        self.assertEqual("field 'bits' has bitshift, this is not supported", str(raises.exception))