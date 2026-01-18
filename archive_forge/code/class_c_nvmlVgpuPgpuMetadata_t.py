from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlVgpuPgpuMetadata_t(_PrintableStructure):
    _fields_ = [('version', c_uint), ('revision', c_uint), ('hostDriverVersion', c_char * NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE), ('pgpuVirtualizationCaps', c_uint), ('reserved', c_uint * 5), ('hostSupportedVgpuRange', c_nvmlVgpuVersion_t), ('opaqueDataSize', c_uint), ('opaqueData', c_char * NVML_VGPU_PGPU_METADATA_OPAQUE_DATA_SIZE)]