from __future__ import annotations
import collections.abc
import tempfile
import sys
import warnings
import operator
import io
import itertools
import functools
import ctypes
import os
import gc
import re
import weakref
import pytest
from contextlib import contextmanager
from numpy.compat import pickle
import pathlib
import builtins
from decimal import Decimal
import mmap
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy.core._rational_tests import rational
from numpy.testing import (
from numpy.testing._private.utils import requires_memory, _no_tracing
from numpy.core.tests._locales import CommaDecimalPointLocale
from numpy.lib.recfunctions import repack_fields
from numpy.core.multiarray import _get_ndarray_c_version
from datetime import timedelta, datetime
from numpy.core._internal import _dtype_from_pep3118
from numpy.testing import IS_PYPY
class TestNewBufferProtocol:
    """ Test PEP3118 buffers """

    def _check_roundtrip(self, obj):
        obj = np.asarray(obj)
        x = memoryview(obj)
        y = np.asarray(x)
        y2 = np.array(x)
        assert_(not y.flags.owndata)
        assert_(y2.flags.owndata)
        assert_equal(y.dtype, obj.dtype)
        assert_equal(y.shape, obj.shape)
        assert_array_equal(obj, y)
        assert_equal(y2.dtype, obj.dtype)
        assert_equal(y2.shape, obj.shape)
        assert_array_equal(obj, y2)

    def test_roundtrip(self):
        x = np.array([1, 2, 3, 4, 5], dtype='i4')
        self._check_roundtrip(x)
        x = np.array([[1, 2], [3, 4]], dtype=np.float64)
        self._check_roundtrip(x)
        x = np.zeros((3, 3, 3), dtype=np.float32)[:, 0, :]
        self._check_roundtrip(x)
        dt = [('a', 'b'), ('b', 'h'), ('c', 'i'), ('d', 'l'), ('dx', 'q'), ('e', 'B'), ('f', 'H'), ('g', 'I'), ('h', 'L'), ('hx', 'Q'), ('i', np.single), ('j', np.double), ('k', np.longdouble), ('ix', np.csingle), ('jx', np.cdouble), ('kx', np.clongdouble), ('l', 'S4'), ('m', 'U4'), ('n', 'V3'), ('o', '?'), ('p', np.half)]
        x = np.array([(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, b'aaaa', 'bbbb', b'xxx', True, 1.0)], dtype=dt)
        self._check_roundtrip(x)
        x = np.array(([[1, 2], [3, 4]],), dtype=[('a', (int, (2, 2)))])
        self._check_roundtrip(x)
        x = np.array([1, 2, 3], dtype='>i2')
        self._check_roundtrip(x)
        x = np.array([1, 2, 3], dtype='<i2')
        self._check_roundtrip(x)
        x = np.array([1, 2, 3], dtype='>i4')
        self._check_roundtrip(x)
        x = np.array([1, 2, 3], dtype='<i4')
        self._check_roundtrip(x)
        x = np.array([1, 2, 3], dtype='>q')
        self._check_roundtrip(x)
        if sys.byteorder == 'little':
            x = np.array([1, 2, 3], dtype='>g')
            assert_raises(ValueError, self._check_roundtrip, x)
            x = np.array([1, 2, 3], dtype='<g')
            self._check_roundtrip(x)
        else:
            x = np.array([1, 2, 3], dtype='>g')
            self._check_roundtrip(x)
            x = np.array([1, 2, 3], dtype='<g')
            assert_raises(ValueError, self._check_roundtrip, x)

    def test_roundtrip_half(self):
        half_list = [1.0, -2.0, 6.5504 * 10 ** 4, 2 ** (-14), 2 ** (-24), 0.0, -0.0, float('+inf'), float('-inf'), 0.333251953125]
        x = np.array(half_list, dtype='>e')
        self._check_roundtrip(x)
        x = np.array(half_list, dtype='<e')
        self._check_roundtrip(x)

    def test_roundtrip_single_types(self):
        for typ in np.sctypeDict.values():
            dtype = np.dtype(typ)
            if dtype.char in 'Mm':
                continue
            if dtype.char == 'V':
                continue
            x = np.zeros(4, dtype=dtype)
            self._check_roundtrip(x)
            if dtype.char not in 'qQgG':
                dt = dtype.newbyteorder('<')
                x = np.zeros(4, dtype=dt)
                self._check_roundtrip(x)
                dt = dtype.newbyteorder('>')
                x = np.zeros(4, dtype=dt)
                self._check_roundtrip(x)

    def test_roundtrip_scalar(self):
        self._check_roundtrip(0)

    def test_invalid_buffer_format(self):
        dt = np.dtype([('a', 'uint16'), ('b', 'M8[s]')])
        a = np.empty(3, dt)
        assert_raises((ValueError, BufferError), memoryview, a)
        assert_raises((ValueError, BufferError), memoryview, np.array(3, 'M8[D]'))

    def test_export_simple_1d(self):
        x = np.array([1, 2, 3, 4, 5], dtype='i')
        y = memoryview(x)
        assert_equal(y.format, 'i')
        assert_equal(y.shape, (5,))
        assert_equal(y.ndim, 1)
        assert_equal(y.strides, (4,))
        assert_equal(y.suboffsets, ())
        assert_equal(y.itemsize, 4)

    def test_export_simple_nd(self):
        x = np.array([[1, 2], [3, 4]], dtype=np.float64)
        y = memoryview(x)
        assert_equal(y.format, 'd')
        assert_equal(y.shape, (2, 2))
        assert_equal(y.ndim, 2)
        assert_equal(y.strides, (16, 8))
        assert_equal(y.suboffsets, ())
        assert_equal(y.itemsize, 8)

    def test_export_discontiguous(self):
        x = np.zeros((3, 3, 3), dtype=np.float32)[:, 0, :]
        y = memoryview(x)
        assert_equal(y.format, 'f')
        assert_equal(y.shape, (3, 3))
        assert_equal(y.ndim, 2)
        assert_equal(y.strides, (36, 4))
        assert_equal(y.suboffsets, ())
        assert_equal(y.itemsize, 4)

    def test_export_record(self):
        dt = [('a', 'b'), ('b', 'h'), ('c', 'i'), ('d', 'l'), ('dx', 'q'), ('e', 'B'), ('f', 'H'), ('g', 'I'), ('h', 'L'), ('hx', 'Q'), ('i', np.single), ('j', np.double), ('k', np.longdouble), ('ix', np.csingle), ('jx', np.cdouble), ('kx', np.clongdouble), ('l', 'S4'), ('m', 'U4'), ('n', 'V3'), ('o', '?'), ('p', np.half)]
        x = np.array([(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, b'aaaa', 'bbbb', b'   ', True, 1.0)], dtype=dt)
        y = memoryview(x)
        assert_equal(y.shape, (1,))
        assert_equal(y.ndim, 1)
        assert_equal(y.suboffsets, ())
        sz = sum([np.dtype(b).itemsize for a, b in dt])
        if np.dtype('l').itemsize == 4:
            assert_equal(y.format, 'T{b:a:=h:b:i:c:l:d:q:dx:B:e:@H:f:=I:g:L:h:Q:hx:f:i:d:j:^g:k:=Zf:ix:Zd:jx:^Zg:kx:4s:l:=4w:m:3x:n:?:o:@e:p:}')
        else:
            assert_equal(y.format, 'T{b:a:=h:b:i:c:q:d:q:dx:B:e:@H:f:=I:g:Q:h:Q:hx:f:i:d:j:^g:k:=Zf:ix:Zd:jx:^Zg:kx:4s:l:=4w:m:3x:n:?:o:@e:p:}')
        if not np.ones(1).strides[0] == np.iinfo(np.intp).max:
            assert_equal(y.strides, (sz,))
        assert_equal(y.itemsize, sz)

    def test_export_subarray(self):
        x = np.array(([[1, 2], [3, 4]],), dtype=[('a', ('i', (2, 2)))])
        y = memoryview(x)
        assert_equal(y.format, 'T{(2,2)i:a:}')
        assert_equal(y.shape, ())
        assert_equal(y.ndim, 0)
        assert_equal(y.strides, ())
        assert_equal(y.suboffsets, ())
        assert_equal(y.itemsize, 16)

    def test_export_endian(self):
        x = np.array([1, 2, 3], dtype='>i')
        y = memoryview(x)
        if sys.byteorder == 'little':
            assert_equal(y.format, '>i')
        else:
            assert_equal(y.format, 'i')
        x = np.array([1, 2, 3], dtype='<i')
        y = memoryview(x)
        if sys.byteorder == 'little':
            assert_equal(y.format, 'i')
        else:
            assert_equal(y.format, '<i')

    def test_export_flags(self):
        assert_raises(ValueError, _multiarray_tests.get_buffer_info, np.arange(5)[::2], ('SIMPLE',))

    @pytest.mark.parametrize(['obj', 'error'], [pytest.param(np.array([1, 2], dtype=rational), ValueError, id='array'), pytest.param(rational(1, 2), TypeError, id='scalar')])
    def test_export_and_pickle_user_dtype(self, obj, error):
        with pytest.raises(error):
            _multiarray_tests.get_buffer_info(obj, ('STRIDED_RO', 'FORMAT'))
        _multiarray_tests.get_buffer_info(obj, ('STRIDED_RO',))
        pickle_obj = pickle.dumps(obj)
        res = pickle.loads(pickle_obj)
        assert_array_equal(res, obj)

    def test_padding(self):
        for j in range(8):
            x = np.array([(1,), (2,)], dtype={'f0': (int, j)})
            self._check_roundtrip(x)

    def test_reference_leak(self):
        if HAS_REFCOUNT:
            count_1 = sys.getrefcount(np.core._internal)
        a = np.zeros(4)
        b = memoryview(a)
        c = np.asarray(b)
        if HAS_REFCOUNT:
            count_2 = sys.getrefcount(np.core._internal)
            assert_equal(count_1, count_2)
        del c

    def test_padded_struct_array(self):
        dt1 = np.dtype([('a', 'b'), ('b', 'i'), ('sub', np.dtype('b,i')), ('c', 'i')], align=True)
        x1 = np.arange(dt1.itemsize, dtype=np.int8).view(dt1)
        self._check_roundtrip(x1)
        dt2 = np.dtype([('a', 'b'), ('b', 'i'), ('c', 'b', (3,)), ('d', 'i')], align=True)
        x2 = np.arange(dt2.itemsize, dtype=np.int8).view(dt2)
        self._check_roundtrip(x2)
        dt3 = np.dtype([('a', 'b'), ('b', 'i'), ('c', 'b'), ('d', 'b'), ('e', 'b'), ('sub', np.dtype('b,i', align=True))])
        x3 = np.arange(dt3.itemsize, dtype=np.int8).view(dt3)
        self._check_roundtrip(x3)

    @pytest.mark.valgrind_error(reason='leaks buffer info cache temporarily.')
    def test_relaxed_strides(self, c=np.ones((1, 10, 10), dtype='i8')):
        c.strides = (-1, 80, 8)
        assert_(memoryview(c).strides == (800, 80, 8))
        fd = io.BytesIO()
        fd.write(c.data)
        fortran = c.T
        assert_(memoryview(fortran).strides == (8, 80, 800))
        arr = np.ones((1, 10))
        if arr.flags.f_contiguous:
            shape, strides = _multiarray_tests.get_buffer_info(arr, ['F_CONTIGUOUS'])
            assert_(strides[0] == 8)
            arr = np.ones((10, 1), order='F')
            shape, strides = _multiarray_tests.get_buffer_info(arr, ['C_CONTIGUOUS'])
            assert_(strides[-1] == 8)

    @pytest.mark.valgrind_error(reason='leaks buffer info cache temporarily.')
    @pytest.mark.skipif(not np.ones((10, 1), order='C').flags.f_contiguous, reason='Test is unnecessary (but fails) without relaxed strides.')
    def test_relaxed_strides_buffer_info_leak(self, arr=np.ones((1, 10))):
        """Test that alternating export of C- and F-order buffers from
        an array which is both C- and F-order when relaxed strides is
        active works.
        This test defines array in the signature to ensure leaking more
        references every time the test is run (catching the leak with
        pytest-leaks).
        """
        for i in range(10):
            _, s = _multiarray_tests.get_buffer_info(arr, ['F_CONTIGUOUS'])
            assert s == (8, 8)
            _, s = _multiarray_tests.get_buffer_info(arr, ['C_CONTIGUOUS'])
            assert s == (80, 8)

    def test_out_of_order_fields(self):
        dt = np.dtype(dict(formats=['<i4', '<i4'], names=['one', 'two'], offsets=[4, 0], itemsize=8))
        arr = np.empty(1, dt)
        with assert_raises(ValueError):
            memoryview(arr)

    def test_max_dims(self):
        a = np.ones((1,) * 32)
        self._check_roundtrip(a)

    @pytest.mark.slow
    def test_error_too_many_dims(self):

        def make_ctype(shape, scalar_type):
            t = scalar_type
            for dim in shape[::-1]:
                t = dim * t
            return t
        c_u8_33d = make_ctype((1,) * 33, ctypes.c_uint8)
        m = memoryview(c_u8_33d())
        assert_equal(m.ndim, 33)
        assert_raises_regex(RuntimeError, 'ndim', np.array, m)
        del c_u8_33d, m
        for i in range(33):
            if gc.collect() == 0:
                break

    def test_error_pointer_type(self):
        m = memoryview(ctypes.pointer(ctypes.c_uint8()))
        assert_('&' in m.format)
        assert_raises_regex(ValueError, 'format string', np.array, m)

    def test_error_message_unsupported(self):
        t = ctypes.c_wchar * 4
        with assert_raises(ValueError) as cm:
            np.array(t())
        exc = cm.exception
        with assert_raises_regex(NotImplementedError, "Unrepresentable .* 'u' \\(UCS-2 strings\\)"):
            raise exc.__cause__

    def test_ctypes_integer_via_memoryview(self):
        for c_integer in {ctypes.c_int, ctypes.c_long, ctypes.c_longlong}:
            value = c_integer(42)
            with warnings.catch_warnings(record=True):
                warnings.filterwarnings('always', '.*\\bctypes\\b', RuntimeWarning)
                np.asarray(value)

    def test_ctypes_struct_via_memoryview(self):

        class foo(ctypes.Structure):
            _fields_ = [('a', ctypes.c_uint8), ('b', ctypes.c_uint32)]
        f = foo(a=1, b=2)
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings('always', '.*\\bctypes\\b', RuntimeWarning)
            arr = np.asarray(f)
        assert_equal(arr['a'], 1)
        assert_equal(arr['b'], 2)
        f.a = 3
        assert_equal(arr['a'], 3)

    @pytest.mark.parametrize('obj', [np.ones(3), np.ones(1, dtype='i,i')[()]])
    def test_error_if_stored_buffer_info_is_corrupted(self, obj):
        """
        If a user extends a NumPy array before 1.20 and then runs it
        on NumPy 1.20+. A C-subclassed array might in theory modify
        the new buffer-info field. This checks that an error is raised
        if this happens (for buffer export), an error is written on delete.
        This is a sanity check to help users transition to safe code, it
        may be deleted at any point.
        """
        _multiarray_tests.corrupt_or_fix_bufferinfo(obj)
        name = type(obj)
        with pytest.raises(RuntimeError, match=f'.*{name} appears to be C subclassed'):
            memoryview(obj)
        _multiarray_tests.corrupt_or_fix_bufferinfo(obj)

    def test_no_suboffsets(self):
        try:
            import _testbuffer
        except ImportError:
            raise pytest.skip('_testbuffer is not available')
        for shape in [(2, 3), (2, 3, 4)]:
            data = list(range(np.prod(shape)))
            buffer = _testbuffer.ndarray(data, shape, format='i', flags=_testbuffer.ND_PIL)
            msg = 'NumPy currently does not support.*suboffsets'
            with pytest.raises(BufferError, match=msg):
                np.asarray(buffer)
            with pytest.raises(BufferError, match=msg):
                np.asarray([buffer])
            with pytest.raises(BufferError):
                np.frombuffer(buffer)