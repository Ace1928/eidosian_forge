from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlGpuInstanceProfileInfo_t(Structure):
    _fields_ = [('id', c_uint), ('isP2pSupported', c_uint), ('sliceCount', c_uint), ('instanceCount', c_uint), ('multiprocessorCount', c_uint), ('copyEngineCount', c_uint), ('decoderCount', c_uint), ('encoderCount', c_uint), ('jpegCount', c_uint), ('ofaCount', c_uint), ('memorySizeMB', c_ulonglong)]