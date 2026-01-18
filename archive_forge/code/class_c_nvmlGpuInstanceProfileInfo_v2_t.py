from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlGpuInstanceProfileInfo_v2_t(_PrintableStructure):
    _fields_ = [('version', c_uint), ('id', c_uint), ('isP2pSupported', c_uint), ('sliceCount', c_uint), ('instanceCount', c_uint), ('multiprocessorCount', c_uint), ('copyEngineCount', c_uint), ('decoderCount', c_uint), ('encoderCount', c_uint), ('jpegCount', c_uint), ('ofaCount', c_uint), ('memorySizeMB', c_ulonglong), ('name', c_char * NVML_DEVICE_NAME_V2_BUFFER_SIZE)]

    def __init__(self):
        super(c_nvmlGpuInstanceProfileInfo_v2_t, self).__init__(version=nvmlGpuInstanceProfileInfo_v2)