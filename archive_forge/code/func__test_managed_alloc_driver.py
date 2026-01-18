import numpy as np
from ctypes import byref, c_size_t
from numba.cuda.cudadrv.driver import device_memset, driver, USE_NV_BINDING
from numba import cuda
from numba.cuda.testing import unittest, ContextResettingTestCase
from numba.cuda.testing import skip_on_cudasim, skip_on_arm
from numba.tests.support import linux_only
def _test_managed_alloc_driver(self, memory_factor, attach_global=True):
    total_mem_size = self.get_total_gpu_memory()
    n_bytes = int(memory_factor * total_mem_size)
    ctx = cuda.current_context()
    mem = ctx.memallocmanaged(n_bytes, attach_global=attach_global)
    dtype = np.dtype(np.uint8)
    n_elems = n_bytes // dtype.itemsize
    ary = np.ndarray(shape=n_elems, dtype=dtype, buffer=mem)
    magic = 171
    device_memset(mem, magic, n_bytes)
    ctx.synchronize()
    self.assertTrue(np.all(ary == magic))