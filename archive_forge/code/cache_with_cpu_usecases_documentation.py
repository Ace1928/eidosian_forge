import sys
from numba import cuda, njit
from numba.cuda.testing import CUDATestCase
from numba.cuda.tests.cudapy.cache_usecases import CUDAUseCase, UseCase

    Tests for functionality of this module's functions.
    Note this does not define any "test_*" method, instead check_module()
    should be called by hand.
    