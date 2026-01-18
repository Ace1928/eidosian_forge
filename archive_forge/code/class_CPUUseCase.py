import sys
from numba import cuda, njit
from numba.cuda.testing import CUDATestCase
from numba.cuda.tests.cudapy.cache_usecases import CUDAUseCase, UseCase
class CPUUseCase(UseCase):

    def _call(self, ret, *args):
        self._func(ret, *args)