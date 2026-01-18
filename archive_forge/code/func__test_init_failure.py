import multiprocessing as mp
import os
from numba import cuda
from numba.cuda.cudadrv.driver import CudaAPIError, driver
from numba.cuda.cudadrv.error import CudaSupportError
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
def _test_init_failure(self, target, expected):
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()
    proc = ctx.Process(target=target, args=(result_queue,))
    proc.start()
    proc.join(30)
    success, msg = result_queue.get()
    if not success:
        self.fail('CudaSupportError not raised')
    self.assertIn(expected, msg)