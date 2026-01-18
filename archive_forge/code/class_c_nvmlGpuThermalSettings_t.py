from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlGpuThermalSettings_t(Structure):
    _fields_ = [('count', c_uint), ('sensor', c_nvmlGpuThermalSensor_t * NVML_MAX_THERMAL_SENSORS_PER_GPU)]