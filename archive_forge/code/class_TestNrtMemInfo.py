import math
import os
import platform
import sys
import re
import numpy as np
from numba import njit
from numba.core import types
from numba.core.runtime import (
from numba.core.extending import intrinsic, include_path
from numba.core.typing import signature
from numba.core.imputils import impl_ret_untracked
from llvmlite import ir
import llvmlite.binding as llvm
from numba.core.unsafe.nrt import NRT_get_api
from numba.tests.support import (EnableNRTStatsMixin, TestCase, temp_directory,
from numba.core.registry import cpu_target
import unittest
class TestNrtMemInfo(unittest.TestCase):
    """
    Unit test for core MemInfo functionality
    """

    def setUp(self):
        Dummy.alive = 0
        rtsys.initialize(cpu_target.target_context)
        super(TestNrtMemInfo, self).setUp()

    def test_meminfo_refct_1(self):
        d = Dummy()
        self.assertEqual(Dummy.alive, 1)
        addr = 3735931646
        mi = rtsys.meminfo_new(addr, d)
        self.assertEqual(mi.refcount, 1)
        del d
        self.assertEqual(Dummy.alive, 1)
        mi.acquire()
        self.assertEqual(mi.refcount, 2)
        self.assertEqual(Dummy.alive, 1)
        mi.release()
        self.assertEqual(mi.refcount, 1)
        del mi
        self.assertEqual(Dummy.alive, 0)

    def test_meminfo_refct_2(self):
        d = Dummy()
        self.assertEqual(Dummy.alive, 1)
        addr = 3735931646
        mi = rtsys.meminfo_new(addr, d)
        self.assertEqual(mi.refcount, 1)
        del d
        self.assertEqual(Dummy.alive, 1)
        for ct in range(100):
            mi.acquire()
        self.assertEqual(mi.refcount, 1 + 100)
        self.assertEqual(Dummy.alive, 1)
        for _ in range(100):
            mi.release()
        self.assertEqual(mi.refcount, 1)
        del mi
        self.assertEqual(Dummy.alive, 0)

    def test_fake_memoryview(self):
        d = Dummy()
        self.assertEqual(Dummy.alive, 1)
        addr = 3735931646
        mi = rtsys.meminfo_new(addr, d)
        self.assertEqual(mi.refcount, 1)
        mview = memoryview(mi)
        self.assertEqual(mi.refcount, 1)
        self.assertEqual(addr, mi.data)
        self.assertFalse(mview.readonly)
        self.assertIs(mi, mview.obj)
        self.assertTrue(mview.c_contiguous)
        self.assertEqual(mview.itemsize, 1)
        self.assertEqual(mview.ndim, 1)
        del d
        del mi
        self.assertEqual(Dummy.alive, 1)
        del mview
        self.assertEqual(Dummy.alive, 0)

    def test_memoryview(self):
        from ctypes import c_uint32, c_void_p, POINTER, cast
        dtype = np.dtype(np.uint32)
        bytesize = dtype.itemsize * 10
        mi = rtsys.meminfo_alloc(bytesize, safe=True)
        addr = mi.data
        c_arr = cast(c_void_p(mi.data), POINTER(c_uint32 * 10))
        for i in range(10):
            self.assertEqual(c_arr.contents[i], 3419130827)
        for i in range(10):
            c_arr.contents[i] = i + 1
        mview = memoryview(mi)
        self.assertEqual(mview.nbytes, bytesize)
        self.assertFalse(mview.readonly)
        self.assertIs(mi, mview.obj)
        self.assertTrue(mview.c_contiguous)
        self.assertEqual(mview.itemsize, 1)
        self.assertEqual(mview.ndim, 1)
        del mi
        arr = np.ndarray(dtype=dtype, shape=mview.nbytes // dtype.itemsize, buffer=mview)
        del mview
        np.testing.assert_equal(np.arange(arr.size) + 1, arr)
        arr += 1
        for i in range(10):
            self.assertEqual(c_arr.contents[i], i + 2)
        self.assertEqual(arr.ctypes.data, addr)
        del arr

    def test_buffer(self):
        from ctypes import c_uint32, c_void_p, POINTER, cast
        dtype = np.dtype(np.uint32)
        bytesize = dtype.itemsize * 10
        mi = rtsys.meminfo_alloc(bytesize, safe=True)
        self.assertEqual(mi.refcount, 1)
        addr = mi.data
        c_arr = cast(c_void_p(addr), POINTER(c_uint32 * 10))
        for i in range(10):
            self.assertEqual(c_arr.contents[i], 3419130827)
        for i in range(10):
            c_arr.contents[i] = i + 1
        arr = np.ndarray(dtype=dtype, shape=bytesize // dtype.itemsize, buffer=mi)
        self.assertEqual(mi.refcount, 1)
        del mi
        np.testing.assert_equal(np.arange(arr.size) + 1, arr)
        arr += 1
        for i in range(10):
            self.assertEqual(c_arr.contents[i], i + 2)
        self.assertEqual(arr.ctypes.data, addr)
        del arr

    @skip_if_32bit
    def test_allocate_invalid_size(self):
        size = types.size_t.maxval // 8 // 2
        for pred in (True, False):
            with self.assertRaises(MemoryError) as raises:
                rtsys.meminfo_alloc(size, safe=pred)
            self.assertIn(f'Requested allocation of {size} bytes failed.', str(raises.exception))

    def test_allocate_negative_size(self):
        size = -10
        for pred in (True, False):
            with self.assertRaises(ValueError) as raises:
                rtsys.meminfo_alloc(size, safe=pred)
            msg = f'Cannot allocate a negative number of bytes: {size}.'
            self.assertIn(msg, str(raises.exception))