import numpy as np
from numba import vectorize, cuda
from numba.tests.npyufunc.test_vectorize_decor import BaseVectorizeDecor, \
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
@skip_on_cudasim('ufunc API unsupported in the simulator')
class TestVectorizeUnrecognizedArg(BaseVectorizeUnrecognizedArg, CUDATestCase):

    def test_target_cuda_unrecognized_arg(self):
        self._test_target_unrecognized_arg('cuda')