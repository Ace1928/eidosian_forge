from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlVgpuSchedulerParams_t(Union):
    _fields_ = [('vgpuSchedDataWithARR', c_nvmlVgpuSchedDataWithARR_t), ('vgpuSchedData', c_nvmlVgpuSchedData_t)]