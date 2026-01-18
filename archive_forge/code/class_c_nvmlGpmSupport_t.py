from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlGpmSupport_t(_PrintableStructure):
    _fields_ = [('version', c_uint), ('isSupportedDevice', c_uint)]