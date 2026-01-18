from ctypes import byref, c_int, c_void_p, sizeof
from numba.cuda.cudadrv.driver import (host_to_device, device_to_host, driver,
from numba.cuda.cudadrv import devices, drvapi, driver as _driver
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
@skip_on_cudasim('CUDA Driver API unsupported in the simulator')
class TestCudaDriver(CUDATestCase):

    def setUp(self):
        super().setUp()
        self.assertTrue(len(devices.gpus) > 0)
        self.context = devices.get_context()
        device = self.context.device
        ccmajor, _ = device.compute_capability
        if ccmajor >= 2:
            self.ptx = ptx2
        else:
            self.ptx = ptx1

    def tearDown(self):
        super().tearDown()
        del self.context

    def test_cuda_driver_basic(self):
        module = self.context.create_module_ptx(self.ptx)
        function = module.get_function('_Z10helloworldPi')
        array = (c_int * 100)()
        memory = self.context.memalloc(sizeof(array))
        host_to_device(memory, array, sizeof(array))
        ptr = memory.device_ctypes_pointer
        stream = 0
        if _driver.USE_NV_BINDING:
            ptr = c_void_p(int(ptr))
            stream = _driver.binding.CUstream(stream)
        launch_kernel(function.handle, 1, 1, 1, 100, 1, 1, 0, stream, [ptr])
        device_to_host(array, memory, sizeof(array))
        for i, v in enumerate(array):
            self.assertEqual(i, v)
        module.unload()

    def test_cuda_driver_stream_operations(self):
        module = self.context.create_module_ptx(self.ptx)
        function = module.get_function('_Z10helloworldPi')
        array = (c_int * 100)()
        stream = self.context.create_stream()
        with stream.auto_synchronize():
            memory = self.context.memalloc(sizeof(array))
            host_to_device(memory, array, sizeof(array), stream=stream)
            ptr = memory.device_ctypes_pointer
            if _driver.USE_NV_BINDING:
                ptr = c_void_p(int(ptr))
            launch_kernel(function.handle, 1, 1, 1, 100, 1, 1, 0, stream.handle, [ptr])
        device_to_host(array, memory, sizeof(array), stream=stream)
        for i, v in enumerate(array):
            self.assertEqual(i, v)

    def test_cuda_driver_default_stream(self):
        ds = self.context.get_default_stream()
        self.assertIn('Default CUDA stream', repr(ds))
        self.assertEqual(0, int(ds))
        self.assertTrue(ds)
        self.assertFalse(ds.external)

    def test_cuda_driver_legacy_default_stream(self):
        ds = self.context.get_legacy_default_stream()
        self.assertIn('Legacy default CUDA stream', repr(ds))
        self.assertEqual(1, int(ds))
        self.assertTrue(ds)
        self.assertFalse(ds.external)

    def test_cuda_driver_per_thread_default_stream(self):
        ds = self.context.get_per_thread_default_stream()
        self.assertIn('Per-thread default CUDA stream', repr(ds))
        self.assertEqual(2, int(ds))
        self.assertTrue(ds)
        self.assertFalse(ds.external)

    def test_cuda_driver_stream(self):
        s = self.context.create_stream()
        self.assertIn('CUDA stream', repr(s))
        self.assertNotIn('Default', repr(s))
        self.assertNotIn('External', repr(s))
        self.assertNotEqual(0, int(s))
        self.assertTrue(s)
        self.assertFalse(s.external)

    def test_cuda_driver_external_stream(self):
        if _driver.USE_NV_BINDING:
            handle = driver.cuStreamCreate(0)
            ptr = int(handle)
        else:
            handle = drvapi.cu_stream()
            driver.cuStreamCreate(byref(handle), 0)
            ptr = handle.value
        s = self.context.create_external_stream(ptr)
        self.assertIn('External CUDA stream', repr(s))
        self.assertNotIn('efault', repr(s))
        self.assertEqual(ptr, int(s))
        self.assertTrue(s)
        self.assertTrue(s.external)

    def test_cuda_driver_occupancy(self):
        module = self.context.create_module_ptx(self.ptx)
        function = module.get_function('_Z10helloworldPi')
        value = self.context.get_active_blocks_per_multiprocessor(function, 128, 128)
        self.assertTrue(value > 0)

        def b2d(bs):
            return bs
        grid, block = self.context.get_max_potential_block_size(function, b2d, 128, 128)
        self.assertTrue(grid > 0)
        self.assertTrue(block > 0)