import ctypes
import numpy as np
from numba.cuda.cudadrv import driver, drvapi, devices
from numba.cuda.testing import unittest, ContextResettingTestCase
from numba.cuda.testing import skip_on_cudasim
class mystruct(ctypes.Structure):
    _fields_ = [('x', ctypes.c_int), ('y', ctypes.c_int)]