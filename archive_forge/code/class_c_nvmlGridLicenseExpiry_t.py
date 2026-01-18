from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlGridLicenseExpiry_t(_PrintableStructure):
    _fields_ = [('year', c_uint32), ('month', c_uint16), ('day', c_uint16), ('hour', c_uint16), ('min', c_uint16), ('sec', c_uint16), ('status', c_uint8)]