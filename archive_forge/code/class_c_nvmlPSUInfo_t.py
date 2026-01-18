from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlPSUInfo_t(_PrintableStructure):
    _fields_ = [('state', c_char * 256), ('current', c_uint), ('voltage', c_uint), ('power', c_uint)]