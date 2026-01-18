import sys
import sysconfig
import weakref
from pathlib import Path
import pytest
import numpy as np
from numpy.ctypeslib import ndpointer, load_library, as_array
from numpy.testing import assert_, assert_array_equal, assert_raises, assert_equal
@pytest.mark.skipif(ctypes is None, reason='ctypes not available on this python installation')
class TestNdpointerCFunc:

    def test_arguments(self):
        """ Test that arguments are coerced from arrays """
        c_forward_pointer.restype = ctypes.c_void_p
        c_forward_pointer.argtypes = (ndpointer(ndim=2),)
        c_forward_pointer(np.zeros((2, 3)))
        assert_raises(ctypes.ArgumentError, c_forward_pointer, np.zeros((2, 3, 4)))

    @pytest.mark.parametrize('dt', [float, np.dtype(dict(formats=['<i4', '<i4'], names=['a', 'b'], offsets=[0, 2], itemsize=6))], ids=['float', 'overlapping-fields'])
    def test_return(self, dt):
        """ Test that return values are coerced to arrays """
        arr = np.zeros((2, 3), dt)
        ptr_type = ndpointer(shape=arr.shape, dtype=arr.dtype)
        c_forward_pointer.restype = ptr_type
        c_forward_pointer.argtypes = (ptr_type,)
        arr2 = c_forward_pointer(arr)
        assert_equal(arr2.dtype, arr.dtype)
        assert_equal(arr2.shape, arr.shape)
        assert_equal(arr2.__array_interface__['data'], arr.__array_interface__['data'])

    def test_vague_return_value(self):
        """ Test that vague ndpointer return values do not promote to arrays """
        arr = np.zeros((2, 3))
        ptr_type = ndpointer(dtype=arr.dtype)
        c_forward_pointer.restype = ptr_type
        c_forward_pointer.argtypes = (ptr_type,)
        ret = c_forward_pointer(arr)
        assert_(isinstance(ret, ptr_type))