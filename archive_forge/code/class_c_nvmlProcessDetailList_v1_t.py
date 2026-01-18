from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlProcessDetailList_v1_t(_PrintableStructure):
    _fields_ = [('version', c_uint), ('mode', _nvmlProcessMode_t), ('numProcArrayEntries', c_uint), ('procArray', POINTER(c_nvmlProcessDetail_v1_t))]
    _fmt_ = {'numProcArrayEntries': '%d B'}