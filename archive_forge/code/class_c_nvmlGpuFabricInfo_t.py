from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlGpuFabricInfo_t(_PrintableStructure):
    _fields_ = [('clusterUuid', c_char * NVML_DEVICE_UUID_BUFFER_SIZE), ('status', _nvmlReturn_t), ('partitionId', c_uint32), ('state', _nvmlGpuFabricState_t)]