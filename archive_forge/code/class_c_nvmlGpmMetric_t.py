from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlGpmMetric_t(_PrintableStructure):
    _fields_ = [('metricId', c_uint), ('nvmlReturn', _nvmlReturn_t), ('value', c_double), ('metricInfo', c_metricInfo_t)]