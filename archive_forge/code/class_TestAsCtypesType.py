import sys
import sysconfig
import weakref
from pathlib import Path
import pytest
import numpy as np
from numpy.ctypeslib import ndpointer, load_library, as_array
from numpy.testing import assert_, assert_array_equal, assert_raises, assert_equal
@pytest.mark.skipif(ctypes is None, reason='ctypes not available on this python installation')
class TestAsCtypesType:
    """ Test conversion from dtypes to ctypes types """

    def test_scalar(self):
        dt = np.dtype('<u2')
        ct = np.ctypeslib.as_ctypes_type(dt)
        assert_equal(ct, ctypes.c_uint16.__ctype_le__)
        dt = np.dtype('>u2')
        ct = np.ctypeslib.as_ctypes_type(dt)
        assert_equal(ct, ctypes.c_uint16.__ctype_be__)
        dt = np.dtype('u2')
        ct = np.ctypeslib.as_ctypes_type(dt)
        assert_equal(ct, ctypes.c_uint16)

    def test_subarray(self):
        dt = np.dtype((np.int32, (2, 3)))
        ct = np.ctypeslib.as_ctypes_type(dt)
        assert_equal(ct, 2 * (3 * ctypes.c_int32))

    def test_structure(self):
        dt = np.dtype([('a', np.uint16), ('b', np.uint32)])
        ct = np.ctypeslib.as_ctypes_type(dt)
        assert_(issubclass(ct, ctypes.Structure))
        assert_equal(ctypes.sizeof(ct), dt.itemsize)
        assert_equal(ct._fields_, [('a', ctypes.c_uint16), ('b', ctypes.c_uint32)])

    def test_structure_aligned(self):
        dt = np.dtype([('a', np.uint16), ('b', np.uint32)], align=True)
        ct = np.ctypeslib.as_ctypes_type(dt)
        assert_(issubclass(ct, ctypes.Structure))
        assert_equal(ctypes.sizeof(ct), dt.itemsize)
        assert_equal(ct._fields_, [('a', ctypes.c_uint16), ('', ctypes.c_char * 2), ('b', ctypes.c_uint32)])

    def test_union(self):
        dt = np.dtype(dict(names=['a', 'b'], offsets=[0, 0], formats=[np.uint16, np.uint32]))
        ct = np.ctypeslib.as_ctypes_type(dt)
        assert_(issubclass(ct, ctypes.Union))
        assert_equal(ctypes.sizeof(ct), dt.itemsize)
        assert_equal(ct._fields_, [('a', ctypes.c_uint16), ('b', ctypes.c_uint32)])

    def test_padded_union(self):
        dt = np.dtype(dict(names=['a', 'b'], offsets=[0, 0], formats=[np.uint16, np.uint32], itemsize=5))
        ct = np.ctypeslib.as_ctypes_type(dt)
        assert_(issubclass(ct, ctypes.Union))
        assert_equal(ctypes.sizeof(ct), dt.itemsize)
        assert_equal(ct._fields_, [('a', ctypes.c_uint16), ('b', ctypes.c_uint32), ('', ctypes.c_char * 5)])

    def test_overlapping(self):
        dt = np.dtype(dict(names=['a', 'b'], offsets=[0, 2], formats=[np.uint32, np.uint32]))
        assert_raises(NotImplementedError, np.ctypeslib.as_ctypes_type, dt)