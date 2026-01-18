import warnings
import itertools
import sys
import ctypes as ct
import pytest
from pytest import param
import numpy as np
import numpy.core._umath_tests as umt
import numpy.linalg._umath_linalg as uml
import numpy.core._operand_flag_tests as opflag_tests
import numpy.core._rational_tests as _rational_tests
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.compat import pickle
class TestLowlevelAPIAccess:

    def test_resolve_dtypes_basic(self):
        i4 = np.dtype('i4')
        f4 = np.dtype('f4')
        f8 = np.dtype('f8')
        r = np.add.resolve_dtypes((i4, f4, None))
        assert r == (f8, f8, f8)
        r = np.add.resolve_dtypes((i4, i4, None), signature=(None, None, 'f4'))
        assert r == (f4, f4, f4)
        r = np.add.resolve_dtypes((f4, int, None))
        assert r == (f4, f4, f4)
        with pytest.raises(TypeError):
            np.add.resolve_dtypes((i4, f4, None), casting='no')

    def test_weird_dtypes(self):
        S0 = np.dtype('S0')
        r = np.equal.resolve_dtypes((S0, S0, None))
        assert r == (S0, S0, np.dtype(bool))
        dts = np.dtype('10i')
        with pytest.raises(TypeError):
            np.equal.resolve_dtypes((dts, dts, None))

    def test_resolve_dtypes_reduction(self):
        i4 = np.dtype('i4')
        with pytest.raises(NotImplementedError):
            np.add.resolve_dtypes((i4, i4, i4), reduction=True)

    @pytest.mark.parametrize('dtypes', [(np.dtype('i'), np.dtype('i')), (None, np.dtype('i'), np.dtype('f')), (np.dtype('i'), None, np.dtype('f')), ('i4', 'i4', None)])
    def test_resolve_dtypes_errors(self, dtypes):
        with pytest.raises(TypeError):
            np.add.resolve_dtypes(dtypes)

    def test_resolve_dtypes_reduction(self):
        i2 = np.dtype('i2')
        long_ = np.dtype('long')
        res = np.add.resolve_dtypes((None, i2, None), reduction=True)
        assert res == (long_, long_, long_)

    def test_resolve_dtypes_reduction_errors(self):
        i2 = np.dtype('i2')
        with pytest.raises(TypeError):
            np.add.resolve_dtypes((None, i2, i2))
        with pytest.raises(TypeError):
            np.add.signature((None, None, 'i4'))

    @pytest.mark.skipif(not hasattr(ct, 'pythonapi'), reason='`ctypes.pythonapi` required for capsule unpacking.')
    def test_loop_access(self):
        data_t = ct.ARRAY(ct.c_char_p, 2)
        dim_t = ct.ARRAY(ct.c_ssize_t, 1)
        strides_t = ct.ARRAY(ct.c_ssize_t, 2)
        strided_loop_t = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, data_t, dim_t, strides_t, ct.c_void_p)

        class call_info_t(ct.Structure):
            _fields_ = [('strided_loop', strided_loop_t), ('context', ct.c_void_p), ('auxdata', ct.c_void_p), ('requires_pyapi', ct.c_byte), ('no_floatingpoint_errors', ct.c_byte)]
        i4 = np.dtype('i4')
        dt, call_info_obj = np.negative._resolve_dtypes_and_context((i4, i4))
        assert dt == (i4, i4)
        np.negative._get_strided_loop(call_info_obj)
        ct.pythonapi.PyCapsule_GetPointer.restype = ct.c_void_p
        call_info = ct.pythonapi.PyCapsule_GetPointer(ct.py_object(call_info_obj), ct.c_char_p(b'numpy_1.24_ufunc_call_info'))
        call_info = ct.cast(call_info, ct.POINTER(call_info_t)).contents
        arr = np.arange(10, dtype=i4)
        call_info.strided_loop(call_info.context, data_t(arr.ctypes.data, arr.ctypes.data), arr.ctypes.shape, strides_t(arr.ctypes.strides[0], arr.ctypes.strides[0]), call_info.auxdata)
        assert_array_equal(arr, -np.arange(10, dtype=i4))

    @pytest.mark.parametrize('strides', [1, (1, 2, 3), (1, '2')])
    def test__get_strided_loop_errors_bad_strides(self, strides):
        i4 = np.dtype('i4')
        dt, call_info = np.negative._resolve_dtypes_and_context((i4, i4))
        with pytest.raises(TypeError, match='fixed_strides.*tuple.*or None'):
            np.negative._get_strided_loop(call_info, fixed_strides=strides)

    def test__get_strided_loop_errors_bad_call_info(self):
        i4 = np.dtype('i4')
        dt, call_info = np.negative._resolve_dtypes_and_context((i4, i4))
        with pytest.raises(ValueError, match='PyCapsule'):
            np.negative._get_strided_loop('not the capsule!')
        with pytest.raises(TypeError, match='.*incompatible context'):
            np.add._get_strided_loop(call_info)
        np.negative._get_strided_loop(call_info)
        with pytest.raises(TypeError):
            np.negative._get_strided_loop(call_info)

    def test_long_arrays(self):
        t = np.zeros((1029, 917), dtype=np.single)
        t[0][0] = 1
        t[28][414] = 1
        tc = np.cos(t)
        assert_equal(tc[0][0], tc[28][414])