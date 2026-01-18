from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlRowRemapperHistogramValues(Structure):
    _fields_ = [('max', c_uint), ('high', c_uint), ('partial', c_uint), ('low', c_uint), ('none', c_uint)]