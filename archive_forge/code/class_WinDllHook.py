import ctypes
import functools
from winappdbg import compat
import sys
class WinDllHook(object):

    def __getattr__(self, name):
        if name.startswith('_'):
            return object.__getattr__(self, name)
        return WinFuncHook(name)