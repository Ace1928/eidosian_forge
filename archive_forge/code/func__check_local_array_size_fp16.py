import numpy as np
from numba import cuda, int32, complex128, void
from numba.core import types
from numba.core.errors import TypingError
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from .extensions_usecases import test_struct_model_type, TestStruct
def _check_local_array_size_fp16(self, shape, expected, ty):

    @cuda.jit
    def s(a):
        arr = cuda.local.array(shape, dtype=ty)
        a[0] = arr.size
    result = np.zeros(1, dtype=np.float16)
    s[1, 1](result)
    self.assertEqual(result[0], expected)