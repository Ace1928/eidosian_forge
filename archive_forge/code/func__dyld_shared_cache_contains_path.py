import os
from ctypes.macholib.framework import framework_info
from ctypes.macholib.dylib import dylib_info
from itertools import *
def _dyld_shared_cache_contains_path(*args):
    raise NotImplementedError