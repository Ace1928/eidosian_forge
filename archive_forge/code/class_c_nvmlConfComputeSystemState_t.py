from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlConfComputeSystemState_t(Structure):
    _fields_ = [('environment', c_uint), ('ccFeature', c_uint), ('devToolsMode', c_uint)]