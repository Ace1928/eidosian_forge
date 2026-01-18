from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlVgpuMetadata_t(_PrintableStructure):
    _fields_ = [('version', c_uint), ('revision', c_uint), ('guestInfoState', _nvmlVgpuGuestInfoState_t), ('guestDriverVersion', c_char * NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE), ('hostDriverVersion', c_char * NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE), ('reserved', c_uint * 6), ('vgpuVirtualizationCaps', c_uint), ('guestVgpuVersion', c_uint), ('opaqueDataSize', c_uint), ('opaqueData', c_char * NVML_VGPU_METADATA_OPAQUE_DATA_SIZE)]