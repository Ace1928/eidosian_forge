import numpy as np
from .common import TestCase, ut
import h5py
from h5py import h5a, h5s, h5t
from h5py import File
from h5py._hl.base import is_empty_dataspace
class TestTypes(BaseAttrs):
    """
        Feature: All supported types can be stored in attributes
    """

    def test_int(self):
        """ Storage of integer types """
        dtypes = (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)
        for dt in dtypes:
            data = np.ndarray((1,), dtype=dt)
            data[...] = 42
            self.f.attrs['x'] = data
            out = self.f.attrs['x']
            self.assertEqual(out.dtype, dt)
            self.assertArrayEqual(out, data)

    def test_float(self):
        """ Storage of floating point types """
        dtypes = tuple((np.dtype(x) for x in ('<f4', '>f4', '>f8', '<f8')))
        for dt in dtypes:
            data = np.ndarray((1,), dtype=dt)
            data[...] = 42.3
            self.f.attrs['x'] = data
            out = self.f.attrs['x']
            print('dtype: ', out.dtype, dt)
            print('value: ', out, data)
            self.assertEqual(out.dtype, dt)
            self.assertArrayEqual(out, data)

    def test_complex(self):
        """ Storage of complex types """
        dtypes = tuple((np.dtype(x) for x in ('<c8', '>c8', '<c16', '>c16')))
        for dt in dtypes:
            data = np.ndarray((1,), dtype=dt)
            data[...] = -4.2j + 35.9
            self.f.attrs['x'] = data
            out = self.f.attrs['x']
            self.assertEqual(out.dtype, dt)
            self.assertArrayEqual(out, data)

    def test_string(self):
        """ Storage of fixed-length strings """
        dtypes = tuple((np.dtype(x) for x in ('|S1', '|S10')))
        for dt in dtypes:
            data = np.ndarray((1,), dtype=dt)
            data[...] = 'h'
            self.f.attrs['x'] = data
            out = self.f.attrs['x']
            self.assertEqual(out.dtype, dt)
            self.assertEqual(out[0], data[0])

    def test_bool(self):
        """ Storage of NumPy booleans """
        data = np.ndarray((2,), dtype=np.bool_)
        data[...] = (True, False)
        self.f.attrs['x'] = data
        out = self.f.attrs['x']
        self.assertEqual(out.dtype, data.dtype)
        self.assertEqual(out[0], data[0])
        self.assertEqual(out[1], data[1])

    def test_vlen_string_array(self):
        """ Storage of vlen byte string arrays"""
        dt = h5py.string_dtype(encoding='ascii')
        data = np.ndarray((2,), dtype=dt)
        data[...] = ('Hello', 'Hi there!  This is HDF5!')
        self.f.attrs['x'] = data
        out = self.f.attrs['x']
        self.assertEqual(out.dtype, dt)
        self.assertEqual(out[0], data[0])
        self.assertEqual(out[1], data[1])

    def test_string_scalar(self):
        """ Storage of variable-length byte string scalars (auto-creation) """
        self.f.attrs['x'] = b'Hello'
        out = self.f.attrs['x']
        self.assertEqual(out, 'Hello')
        self.assertEqual(type(out), str)
        aid = h5py.h5a.open(self.f.id, b'x')
        tid = aid.get_type()
        self.assertEqual(type(tid), h5py.h5t.TypeStringID)
        self.assertEqual(tid.get_cset(), h5py.h5t.CSET_ASCII)
        self.assertTrue(tid.is_variable_str())

    def test_unicode_scalar(self):
        """ Storage of variable-length unicode strings (auto-creation) """
        self.f.attrs['x'] = u'Hello' + chr(9024) + u'!!'
        out = self.f.attrs['x']
        self.assertEqual(out, u'Hello' + chr(9024) + u'!!')
        self.assertEqual(type(out), str)
        aid = h5py.h5a.open(self.f.id, b'x')
        tid = aid.get_type()
        self.assertEqual(type(tid), h5py.h5t.TypeStringID)
        self.assertEqual(tid.get_cset(), h5py.h5t.CSET_UTF8)
        self.assertTrue(tid.is_variable_str())