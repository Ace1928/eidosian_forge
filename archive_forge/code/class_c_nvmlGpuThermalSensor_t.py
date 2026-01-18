from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlGpuThermalSensor_t(Structure):
    _fields_ = [('controller', c_int), ('defaultMinTemp', c_int), ('defaultMaxTemp', c_int), ('currentTemp', c_int), ('target', c_int)]