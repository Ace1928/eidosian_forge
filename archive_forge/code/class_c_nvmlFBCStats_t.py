from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlFBCStats_t(Structure):
    _fields_ = [('sessionsCount', c_uint), ('averageFPS', c_uint), ('averageLatency', c_uint)]