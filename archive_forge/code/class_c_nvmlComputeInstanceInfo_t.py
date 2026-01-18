from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlComputeInstanceInfo_t(Structure):
    _fields_ = [('device', c_nvmlDevice_t), ('gpuInstance', c_nvmlGpuInstance_t), ('id', c_uint), ('profileId', c_uint), ('placement', c_nvmlComputeInstancePlacement_t)]