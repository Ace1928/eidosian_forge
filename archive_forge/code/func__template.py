import ctypes
import numpy as np
from numba.cuda.cudadrv import driver, drvapi, devices
from numba.cuda.testing import unittest, ContextResettingTestCase
from numba.cuda.testing import skip_on_cudasim
def _template(self, obj):
    self.assertTrue(driver.is_device_memory(obj))
    driver.require_device_memory(obj)
    if driver.USE_NV_BINDING:
        expected_class = driver.binding.CUdeviceptr
    else:
        expected_class = drvapi.cu_device_ptr
    self.assertTrue(isinstance(obj.device_ctypes_pointer, expected_class))