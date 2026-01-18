from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlProcessDetail_v1_t(Structure):
    _fields_ = [('pid', c_uint), ('usedGpuMemory', c_ulonglong), ('gpuInstanceId', c_uint), ('computeInstanceId', c_uint), ('usedGpuCcProtectedMemory', c_ulonglong)]