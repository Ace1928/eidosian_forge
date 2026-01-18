from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlExcludedDeviceInfo_t(_PrintableStructure):
    _fields_ = [('pci', nvmlPciInfo_t), ('uuid', c_char * NVML_DEVICE_UUID_BUFFER_SIZE)]