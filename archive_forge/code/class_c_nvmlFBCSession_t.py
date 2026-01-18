from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlFBCSession_t(_PrintableStructure):
    _fields_ = [('sessionId', c_uint), ('pid', c_uint), ('vgpuInstance', _nvmlVgpuInstance_t), ('displayOrdinal', c_uint), ('sessionType', c_uint), ('sessionFlags', c_uint), ('hMaxResolution', c_uint), ('vMaxResolution', c_uint), ('hResolution', c_uint), ('vResolution', c_uint), ('averageFPS', c_uint), ('averageLatency', c_uint)]