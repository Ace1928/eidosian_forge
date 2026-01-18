import numpy as np
import platform
from numba import cuda
from numba.cuda.testing import unittest, ContextResettingTestCase
def _run_copies(self, A):
    A0 = np.copy(A)
    stream = cuda.stream()
    ptr = cuda.to_device(A, copy=False, stream=stream)
    ptr.copy_to_device(A, stream=stream)
    ptr.copy_to_host(A, stream=stream)
    stream.synchronize()
    self.assertTrue(np.allclose(A, A0))