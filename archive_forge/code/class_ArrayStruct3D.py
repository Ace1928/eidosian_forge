from ctypes import *
import numpy as np
import unittest
from numba import _helperlib
class ArrayStruct3D(Structure):
    _fields_ = [('meminfo', c_void_p), ('parent', c_void_p), ('nitems', c_ssize_t), ('itemsize', c_ssize_t), ('data', c_void_p), ('shape', c_ssize_t * 3), ('strides', c_ssize_t * 3)]