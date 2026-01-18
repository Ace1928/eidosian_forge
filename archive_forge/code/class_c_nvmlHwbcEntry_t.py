from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlHwbcEntry_t(_PrintableStructure):
    _fields_ = [('hwbcId', c_uint), ('firmwareVersion', c_char * 32)]