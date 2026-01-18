import ctypes
import numpy as np
import weakref
from numba import cuda
from numba.core import config
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.tests.support import linux_only
class BadVersionEMMPlugin(DeviceOnlyEMMPlugin):
    """A plugin that claims to implement a different interface version"""

    @property
    def interface_version(self):
        return 2