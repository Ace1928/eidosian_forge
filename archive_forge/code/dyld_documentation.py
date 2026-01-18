import os
from ctypes.macholib.framework import framework_info
from ctypes.macholib.dylib import dylib_info
from itertools import *

    Find a framework using dyld semantics in a very loose manner.

    Will take input such as:
        Python
        Python.framework
        Python.framework/Versions/Current
    