from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlComputeInstanceProfileInfo_v2_t(_PrintableStructure):
    _fields_ = [('version', c_uint), ('id', c_uint), ('sliceCount', c_uint), ('instanceCount', c_uint), ('multiprocessorCount', c_uint), ('sharedCopyEngineCount', c_uint), ('sharedDecoderCount', c_uint), ('sharedEncoderCount', c_uint), ('sharedJpegCount', c_uint), ('sharedOfaCount', c_uint), ('name', c_char * NVML_DEVICE_NAME_V2_BUFFER_SIZE)]

    def __init__(self):
        super(c_nvmlComputeInstanceProfileInfo_v2_t, self).__init__(version=nvmlComputeInstanceProfileInfo_v2)