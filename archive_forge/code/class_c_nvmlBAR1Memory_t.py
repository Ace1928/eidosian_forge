from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlBAR1Memory_t(_PrintableStructure):
    _fields_ = [('bar1Total', c_ulonglong), ('bar1Free', c_ulonglong), ('bar1Used', c_ulonglong)]
    _fmt_ = {'<default>': '%d B'}