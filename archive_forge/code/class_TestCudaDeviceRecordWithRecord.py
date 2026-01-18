import numpy as np
import ctypes
from numba.cuda.cudadrv.devicearray import (DeviceRecord, from_record_like,
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
from numba.np import numpy_support
from numba import cuda
class TestCudaDeviceRecordWithRecord(TestCudaDeviceRecord):
    """
    Tests the DeviceRecord class with np.record host types
    """

    def setUp(self):
        CUDATestCase.setUp(self)
        self._create_data(np.recarray)