from numba import cuda, int32, float64, void
from numba.core.errors import TypingError
from numba.core import types
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
import numpy as np
from numba.np import numpy_support as nps
from .extensions_usecases import test_struct_model_type, TestStruct
def _check_shared_array_size_fp16(self, shape, expected, ty):

    @cuda.jit
    def s(a):
        arr = cuda.shared.array(shape, dtype=ty)
        a[0] = arr.size
    result = np.zeros(1, dtype=np.float16)
    s[1, 1](result)
    self.assertEqual(result[0], expected)