import ctypes
import ctypes.util
import os
import sys
import threading
import time
class timespec(ctypes.Structure):
    """Time specification, as described in clock_gettime(3)."""
    _fields_ = (('tv_sec', ctypes.c_long), ('tv_nsec', ctypes.c_long))