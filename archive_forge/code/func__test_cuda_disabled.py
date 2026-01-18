import multiprocessing as mp
import os
from numba import cuda
from numba.cuda.cudadrv.driver import CudaAPIError, driver
from numba.cuda.cudadrv.error import CudaSupportError
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
def _test_cuda_disabled(self, target):
    cuda_disabled = os.environ.get('NUMBA_DISABLE_CUDA')
    os.environ['NUMBA_DISABLE_CUDA'] = '1'
    try:
        expected = 'CUDA is disabled due to setting NUMBA_DISABLE_CUDA=1'
        self._test_init_failure(cuda_disabled_test, expected)
    finally:
        if cuda_disabled is not None:
            os.environ['NUMBA_DISABLE_CUDA'] = cuda_disabled
        else:
            os.environ.pop('NUMBA_DISABLE_CUDA')