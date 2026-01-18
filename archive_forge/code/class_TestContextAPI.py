import numbers
from ctypes import byref
import weakref
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.cuda.cudadrv import driver
class TestContextAPI(CUDATestCase):

    def tearDown(self):
        super().tearDown()
        cuda.close()

    def test_context_memory(self):
        try:
            mem = cuda.current_context().get_memory_info()
        except NotImplementedError:
            self.skipTest('EMM Plugin does not implement get_memory_info()')
        self.assertIsInstance(mem.free, numbers.Number)
        self.assertEqual(mem.free, mem[0])
        self.assertIsInstance(mem.total, numbers.Number)
        self.assertEqual(mem.total, mem[1])
        self.assertLessEqual(mem.free, mem.total)

    @unittest.skipIf(len(cuda.gpus) < 2, 'need more than 1 gpus')
    @skip_on_cudasim('CUDA HW required')
    def test_forbidden_context_switch(self):

        @cuda.require_context
        def switch_gpu():
            with cuda.gpus[1]:
                pass
        with cuda.gpus[0]:
            with self.assertRaises(RuntimeError) as raises:
                switch_gpu()
            self.assertIn('Cannot switch CUDA-context.', str(raises.exception))

    @unittest.skipIf(len(cuda.gpus) < 2, 'need more than 1 gpus')
    def test_accepted_context_switch(self):

        def switch_gpu():
            with cuda.gpus[1]:
                return cuda.current_context().device.id
        with cuda.gpus[0]:
            devid = switch_gpu()
        self.assertEqual(int(devid), 1)