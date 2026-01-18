from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlBridgeChipHierarchy_t(_PrintableStructure):
    _fields_ = [('bridgeCount', c_uint), ('bridgeChipInfo', c_nvmlBridgeChipInfo_t * 128)]