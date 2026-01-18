from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlUnitFanInfo_t(_PrintableStructure):
    _fields_ = [('speed', c_uint), ('state', _nvmlFanState_t)]