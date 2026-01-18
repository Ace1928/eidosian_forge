from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlLedState_t(_PrintableStructure):
    _fields_ = [('cause', c_char * 256), ('color', _nvmlLedColor_t)]