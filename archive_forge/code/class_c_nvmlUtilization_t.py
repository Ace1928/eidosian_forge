from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlUtilization_t(_PrintableStructure):
    _fields_ = [('gpu', c_uint), ('memory', c_uint)]
    _fmt_ = {'<default>': '%d %%'}