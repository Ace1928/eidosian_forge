from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlVgpuLicenseInfo_t(_PrintableStructure):
    _fields_ = [('isLicensed', c_uint8), ('licenseExpiry', c_nvmlVgpuLicenseExpiry_t), ('currentState', c_uint)]