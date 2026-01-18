from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlVgpuPgpuCompatibility_t(Structure):
    _fields_ = [('vgpuVmCompatibility', _nvmlVgpuVmCompatibility_t), ('compatibilityLimitCode', _nvmlVgpuPgpuCompatibilityLimitCode_t)]