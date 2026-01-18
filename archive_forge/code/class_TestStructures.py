from os import path
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from scipy.io import readsav
from scipy.io import _idl
class TestStructures:

    def test_scalars(self):
        s = readsav(path.join(DATA_PATH, 'struct_scalars.sav'), verbose=False)
        assert_identical(s.scalars.a, np.array(np.int16(1)))
        assert_identical(s.scalars.b, np.array(np.int32(2)))
        assert_identical(s.scalars.c, np.array(np.float32(3.0)))
        assert_identical(s.scalars.d, np.array(np.float64(4.0)))
        assert_identical(s.scalars.e, np.array([b'spam'], dtype=object))
        assert_identical(s.scalars.f, np.array(np.complex64(-1.0 + 3j)))

    def test_scalars_replicated(self):
        s = readsav(path.join(DATA_PATH, 'struct_scalars_replicated.sav'), verbose=False)
        assert_identical(s.scalars_rep.a, np.repeat(np.int16(1), 5))
        assert_identical(s.scalars_rep.b, np.repeat(np.int32(2), 5))
        assert_identical(s.scalars_rep.c, np.repeat(np.float32(3.0), 5))
        assert_identical(s.scalars_rep.d, np.repeat(np.float64(4.0), 5))
        assert_identical(s.scalars_rep.e, np.repeat(b'spam', 5).astype(object))
        assert_identical(s.scalars_rep.f, np.repeat(np.complex64(-1.0 + 3j), 5))

    def test_scalars_replicated_3d(self):
        s = readsav(path.join(DATA_PATH, 'struct_scalars_replicated_3d.sav'), verbose=False)
        assert_identical(s.scalars_rep.a, np.repeat(np.int16(1), 24).reshape(4, 3, 2))
        assert_identical(s.scalars_rep.b, np.repeat(np.int32(2), 24).reshape(4, 3, 2))
        assert_identical(s.scalars_rep.c, np.repeat(np.float32(3.0), 24).reshape(4, 3, 2))
        assert_identical(s.scalars_rep.d, np.repeat(np.float64(4.0), 24).reshape(4, 3, 2))
        assert_identical(s.scalars_rep.e, np.repeat(b'spam', 24).reshape(4, 3, 2).astype(object))
        assert_identical(s.scalars_rep.f, np.repeat(np.complex64(-1.0 + 3j), 24).reshape(4, 3, 2))

    def test_arrays(self):
        s = readsav(path.join(DATA_PATH, 'struct_arrays.sav'), verbose=False)
        assert_array_identical(s.arrays.a[0], np.array([1, 2, 3], dtype=np.int16))
        assert_array_identical(s.arrays.b[0], np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32))
        assert_array_identical(s.arrays.c[0], np.array([np.complex64(1 + 2j), np.complex64(7 + 8j)]))
        assert_array_identical(s.arrays.d[0], np.array([b'cheese', b'bacon', b'spam'], dtype=object))

    def test_arrays_replicated(self):
        s = readsav(path.join(DATA_PATH, 'struct_arrays_replicated.sav'), verbose=False)
        assert_(s.arrays_rep.a.dtype.type is np.object_)
        assert_(s.arrays_rep.b.dtype.type is np.object_)
        assert_(s.arrays_rep.c.dtype.type is np.object_)
        assert_(s.arrays_rep.d.dtype.type is np.object_)
        assert_equal(s.arrays_rep.a.shape, (5,))
        assert_equal(s.arrays_rep.b.shape, (5,))
        assert_equal(s.arrays_rep.c.shape, (5,))
        assert_equal(s.arrays_rep.d.shape, (5,))
        for i in range(5):
            assert_array_identical(s.arrays_rep.a[i], np.array([1, 2, 3], dtype=np.int16))
            assert_array_identical(s.arrays_rep.b[i], np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32))
            assert_array_identical(s.arrays_rep.c[i], np.array([np.complex64(1 + 2j), np.complex64(7 + 8j)]))
            assert_array_identical(s.arrays_rep.d[i], np.array([b'cheese', b'bacon', b'spam'], dtype=object))

    def test_arrays_replicated_3d(self):
        s = readsav(path.join(DATA_PATH, 'struct_arrays_replicated_3d.sav'), verbose=False)
        assert_(s.arrays_rep.a.dtype.type is np.object_)
        assert_(s.arrays_rep.b.dtype.type is np.object_)
        assert_(s.arrays_rep.c.dtype.type is np.object_)
        assert_(s.arrays_rep.d.dtype.type is np.object_)
        assert_equal(s.arrays_rep.a.shape, (4, 3, 2))
        assert_equal(s.arrays_rep.b.shape, (4, 3, 2))
        assert_equal(s.arrays_rep.c.shape, (4, 3, 2))
        assert_equal(s.arrays_rep.d.shape, (4, 3, 2))
        for i in range(4):
            for j in range(3):
                for k in range(2):
                    assert_array_identical(s.arrays_rep.a[i, j, k], np.array([1, 2, 3], dtype=np.int16))
                    assert_array_identical(s.arrays_rep.b[i, j, k], np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32))
                    assert_array_identical(s.arrays_rep.c[i, j, k], np.array([np.complex64(1 + 2j), np.complex64(7 + 8j)]))
                    assert_array_identical(s.arrays_rep.d[i, j, k], np.array([b'cheese', b'bacon', b'spam'], dtype=object))

    def test_inheritance(self):
        s = readsav(path.join(DATA_PATH, 'struct_inherit.sav'), verbose=False)
        assert_identical(s.fc.x, np.array([0], dtype=np.int16))
        assert_identical(s.fc.y, np.array([0], dtype=np.int16))
        assert_identical(s.fc.r, np.array([0], dtype=np.int16))
        assert_identical(s.fc.c, np.array([4], dtype=np.int16))

    def test_arrays_corrupt_idl80(self):
        with suppress_warnings() as sup:
            sup.filter(UserWarning, 'Not able to verify number of bytes from header')
            s = readsav(path.join(DATA_PATH, 'struct_arrays_byte_idl80.sav'), verbose=False)
        assert_identical(s.y.x[0], np.array([55, 66], dtype=np.uint8))