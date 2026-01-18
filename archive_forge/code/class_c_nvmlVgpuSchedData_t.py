from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlVgpuSchedData_t(_PrintableStructure):
    _fields_ = [('timeslice', c_uint)]