import ctypes
import numpy as np
from numba.cuda.cudadrv import driver, drvapi, devices
from numba.cuda.testing import unittest, ContextResettingTestCase
from numba.cuda.testing import skip_on_cudasim
@skip_on_cudasim('CUDA Memory API unsupported in the simulator')
class TestMVExtent(ContextResettingTestCase):

    def test_c_contiguous_array(self):
        ary = np.arange(100)
        arysz = ary.dtype.itemsize * ary.size
        s, e = driver.host_memory_extents(ary)
        self.assertTrue(ary.ctypes.data == s)
        self.assertTrue(arysz == driver.host_memory_size(ary))

    def test_f_contiguous_array(self):
        ary = np.asfortranarray(np.arange(100).reshape(2, 50))
        arysz = ary.dtype.itemsize * np.prod(ary.shape)
        s, e = driver.host_memory_extents(ary)
        self.assertTrue(ary.ctypes.data == s)
        self.assertTrue(arysz == driver.host_memory_size(ary))

    def test_single_element_array(self):
        ary = np.asarray(np.uint32(1234))
        arysz = ary.dtype.itemsize
        s, e = driver.host_memory_extents(ary)
        self.assertTrue(ary.ctypes.data == s)
        self.assertTrue(arysz == driver.host_memory_size(ary))

    def test_ctypes_struct(self):

        class mystruct(ctypes.Structure):
            _fields_ = [('x', ctypes.c_int), ('y', ctypes.c_int)]
        data = mystruct(x=123, y=432)
        sz = driver.host_memory_size(data)
        self.assertTrue(ctypes.sizeof(data) == sz)

    def test_ctypes_double(self):
        data = ctypes.c_double(1.234)
        sz = driver.host_memory_size(data)
        self.assertTrue(ctypes.sizeof(data) == sz)