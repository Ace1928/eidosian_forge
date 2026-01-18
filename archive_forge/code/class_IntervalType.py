from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import numpy as np
from numba import config, cuda, njit, types
class IntervalType(types.Type):

    def __init__(self):
        super().__init__(name='Interval')