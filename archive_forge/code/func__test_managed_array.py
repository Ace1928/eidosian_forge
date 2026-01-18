import numpy as np
from ctypes import byref, c_size_t
from numba.cuda.cudadrv.driver import device_memset, driver, USE_NV_BINDING
from numba import cuda
from numba.cuda.testing import unittest, ContextResettingTestCase
from numba.cuda.testing import skip_on_cudasim, skip_on_arm
from numba.tests.support import linux_only
def _test_managed_array(self, attach_global=True):
    ary = cuda.managed_array(100, dtype=np.double)
    ary.fill(123.456)
    self.assertTrue(all(ary == 123.456))

    @cuda.jit('void(double[:])')
    def kernel(x):
        i = cuda.grid(1)
        if i < x.shape[0]:
            x[i] = 1.0
    kernel[10, 10](ary)
    cuda.current_context().synchronize()
    self.assertTrue(all(ary == 1.0))