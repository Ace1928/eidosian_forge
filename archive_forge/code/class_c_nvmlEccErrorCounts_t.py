from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlEccErrorCounts_t(_PrintableStructure):
    _fields_ = [('l1Cache', c_ulonglong), ('l2Cache', c_ulonglong), ('deviceMemory', c_ulonglong), ('registerFile', c_ulonglong)]