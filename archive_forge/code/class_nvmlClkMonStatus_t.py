from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class nvmlClkMonStatus_t(Structure):
    _fields_ = [('bGlobalStatus', c_uint), ('clkMonListSize', c_uint), ('clkMonList', nvmlClkMonFaultInfo_t)]