from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlVgpuSchedulerSetState_t(_PrintableStructure):
    _fields_ = [('schedulerPolicy', c_uint), ('enableARRMode', c_uint), ('schedulerParams', c_nvmlVgpuSchedulerSetParams_t)]