from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlFieldValue_t(_PrintableStructure):
    _fields_ = [('fieldId', c_uint32), ('scopeId', c_uint32), ('timestamp', c_int64), ('latencyUsec', c_int64), ('valueType', _nvmlValueType_t), ('nvmlReturn', _nvmlReturn_t), ('value', c_nvmlValue_t)]