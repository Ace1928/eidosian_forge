import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
class TestRecordDtypeMakeCStruct(TestCase):

    def test_two_scalars(self):

        class Ref(ctypes.Structure):
            _fields_ = [('apple', ctypes.c_int32), ('orange', ctypes.c_float)]
        ty = types.Record.make_c_struct([('apple', types.int32), ('orange', types.float32)])
        self.assertEqual(len(ty), 2)
        self.assertEqual(ty.offset('apple'), Ref.apple.offset)
        self.assertEqual(ty.offset('orange'), Ref.orange.offset)
        self.assertEqual(ty.size, ctypes.sizeof(Ref))
        dtype = ty.dtype
        self.assertTrue(dtype.isalignedstruct)

    def test_three_scalars(self):

        class Ref(ctypes.Structure):
            _fields_ = [('apple', ctypes.c_int32), ('mango', ctypes.c_int8), ('orange', ctypes.c_float)]
        ty = types.Record.make_c_struct([('apple', types.int32), ('mango', types.int8), ('orange', types.float32)])
        self.assertEqual(len(ty), 3)
        self.assertEqual(ty.offset('apple'), Ref.apple.offset)
        self.assertEqual(ty.offset('mango'), Ref.mango.offset)
        self.assertEqual(ty.offset('orange'), Ref.orange.offset)
        self.assertEqual(ty.size, ctypes.sizeof(Ref))
        dtype = ty.dtype
        self.assertTrue(dtype.isalignedstruct)

    def test_complex_struct(self):

        class Complex(ctypes.Structure):
            _fields_ = [('real', ctypes.c_double), ('imag', ctypes.c_double)]

        class Ref(ctypes.Structure):
            _fields_ = [('apple', ctypes.c_int32), ('mango', Complex)]
        ty = types.Record.make_c_struct([('apple', types.intc), ('mango', types.complex128)])
        self.assertEqual(len(ty), 2)
        self.assertEqual(ty.offset('apple'), Ref.apple.offset)
        self.assertEqual(ty.offset('mango'), Ref.mango.offset)
        self.assertEqual(ty.size, ctypes.sizeof(Ref))
        self.assertTrue(ty.dtype.isalignedstruct)

    def test_nestedarray_issue_8132(self):
        data = np.arange(27 * 2, dtype=np.float64).reshape(27, 2)
        recty = types.Record.make_c_struct([('data', types.NestedArray(dtype=types.float64, shape=data.shape))])
        arr = np.array((data,), dtype=recty.dtype)
        [extracted_array] = arr.tolist()
        np.testing.assert_array_equal(extracted_array, data)