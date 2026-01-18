from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class nvmlNvLinkUtilizationControl_t(_PrintableStructure):
    _fields_ = [('units', _nvmlNvLinkUtilizationCountUnits_t), ('pktfilter', _nvmlNvLinkUtilizationCountPktTypes_t)]