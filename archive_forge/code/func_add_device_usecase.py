import numpy as np
import threading
from numba import boolean, config, cuda, float32, float64, int32, int64, void
from numba.core.errors import TypingError
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import math
def add_device_usecase(self, sigs):
    add_device = cuda.jit(sigs, device=True)(add)

    @cuda.jit
    def f(r, x, y):
        r[0] = add_device(x, y)
    return f