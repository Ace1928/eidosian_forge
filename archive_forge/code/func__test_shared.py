from numba import cuda, int32, float64, void
from numba.core.errors import TypingError
from numba.core import types
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
import numpy as np
from numba.np import numpy_support as nps
from .extensions_usecases import test_struct_model_type, TestStruct
def _test_shared(self, arr):
    nelem = len(arr)
    nthreads = 16
    nblocks = int(nelem / nthreads)
    dt = nps.from_dtype(arr.dtype)

    @cuda.jit
    def use_sm_chunk_copy(x, y):
        sm = cuda.shared.array(nthreads, dtype=dt)
        tx = cuda.threadIdx.x
        bx = cuda.blockIdx.x
        bd = cuda.blockDim.x
        i = bx * bd + tx
        if i < len(x):
            sm[tx] = x[i]
        cuda.syncthreads()
        if tx == 0:
            for j in range(nthreads):
                y[bd * bx + j] = sm[j]
    d_result = cuda.device_array_like(arr)
    use_sm_chunk_copy[nblocks, nthreads](arr, d_result)
    host_result = d_result.copy_to_host()
    np.testing.assert_array_equal(arr, host_result)