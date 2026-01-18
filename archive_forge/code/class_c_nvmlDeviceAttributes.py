from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlDeviceAttributes(Structure):
    _fields_ = [('multiprocessorCount', c_uint), ('sharedCopyEngineCount', c_uint), ('sharedDecoderCount', c_uint), ('sharedEncoderCount', c_uint), ('sharedJpegCount', c_uint), ('sharedOfaCount', c_uint), ('gpuInstanceSliceCount', c_uint), ('computeInstanceSliceCount', c_uint), ('memorySizeMB', c_ulonglong)]