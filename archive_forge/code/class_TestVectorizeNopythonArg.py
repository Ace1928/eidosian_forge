import numpy as np
from numba import vectorize, cuda
from numba.tests.npyufunc.test_vectorize_decor import BaseVectorizeDecor, \
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
@skip_on_cudasim('ufunc API unsupported in the simulator')
class TestVectorizeNopythonArg(BaseVectorizeNopythonArg, CUDATestCase):

    def test_target_cuda_nopython(self):
        warnings = ['nopython kwarg for cuda target is redundant']
        self._test_target_nopython('cuda', warnings)