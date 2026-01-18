import numpy as np
from numba import cuda
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support
def _createSampleArrays(self):
    self.sample1d = np.zeros(3, dtype=recordtype)
    self.samplerec1darr = np.zeros(1, dtype=recordwitharray)[0]
    self.samplerec2darr = np.zeros(1, dtype=recordwith2darray)[0]