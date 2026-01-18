import multiprocessing
import os
from numba.core import config
from numba.cuda.cudadrv.runtime import runtime
from numba.cuda.testing import unittest, SerialMixin, skip_on_cudasim
from unittest.mock import patch
class TestVisibleDevices(unittest.TestCase, SerialMixin):

    def test_visible_devices_set_after_import(self):
        from numba import cuda
        if len(cuda.gpus.lst) in (0, 1):
            self.skipTest('This test requires multiple GPUs')
        if os.environ.get('CUDA_VISIBLE_DEVICES'):
            msg = 'Cannot test when CUDA_VISIBLE_DEVICES already set'
            self.skipTest(msg)
        ctx = multiprocessing.get_context('spawn')
        q = ctx.Queue()
        p = ctx.Process(target=set_visible_devices_and_check, args=(q,))
        p.start()
        try:
            visible_gpu_count = q.get()
        finally:
            p.join()
        msg = 'Error running set_visible_devices_and_check'
        self.assertNotEqual(visible_gpu_count, -1, msg=msg)
        self.assertEqual(visible_gpu_count, 1)