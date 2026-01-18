import ctypes
import numpy as np
import weakref
from numba import cuda
from numba.core import config
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.tests.support import linux_only
@skip_on_cudasim('EMM Plugins not supported on CUDA simulator')
class TestDeviceOnlyEMMPlugin(CUDATestCase):
    """
    Tests that the API of an EMM Plugin that implements device allocations
    only is used correctly by Numba.
    """

    def setUp(self):
        super().setUp()
        cuda.close()
        cuda.set_memory_manager(DeviceOnlyEMMPlugin)

    def tearDown(self):
        super().tearDown()
        cuda.close()
        cuda.cudadrv.driver._memory_manager = None

    def test_memalloc(self):
        mgr = cuda.current_context().memory_manager
        arr_1 = np.arange(10)
        d_arr_1 = cuda.device_array_like(arr_1)
        self.assertTrue(mgr.memalloc_called)
        self.assertEqual(mgr.count, 1)
        self.assertEqual(mgr.allocations[1], arr_1.nbytes)
        arr_2 = np.arange(5)
        d_arr_2 = cuda.device_array_like(arr_2)
        self.assertEqual(mgr.count, 2)
        self.assertEqual(mgr.allocations[2], arr_2.nbytes)
        del d_arr_1
        self.assertNotIn(1, mgr.allocations)
        self.assertIn(2, mgr.allocations)
        del d_arr_2
        self.assertNotIn(2, mgr.allocations)

    def test_initialized_in_context(self):
        self.assertTrue(cuda.current_context().memory_manager.initialized)

    def test_reset(self):
        ctx = cuda.current_context()
        ctx.reset()
        self.assertTrue(ctx.memory_manager.reset_called)

    def test_get_memory_info(self):
        ctx = cuda.current_context()
        meminfo = ctx.get_memory_info()
        self.assertTrue(ctx.memory_manager.get_memory_info_called)
        self.assertEqual(meminfo.free, 32)
        self.assertEqual(meminfo.total, 64)

    @linux_only
    def test_get_ipc_handle(self):
        arr = np.arange(2)
        d_arr = cuda.device_array_like(arr)
        ipch = d_arr.get_ipc_handle()
        ctx = cuda.current_context()
        self.assertTrue(ctx.memory_manager.get_ipc_handle_called)
        self.assertIn('Dummy IPC handle for alloc 1', ipch._ipc_handle)