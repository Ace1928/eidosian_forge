from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlEventData_t(_PrintableStructure):
    _fields_ = [('device', c_nvmlDevice_t), ('eventType', c_ulonglong), ('eventData', c_ulonglong), ('gpuInstanceId', c_uint), ('computeInstanceId', c_uint)]
    _fmt_ = {'eventType': '0x%08X'}