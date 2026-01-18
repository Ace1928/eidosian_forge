import unittest
from ctypes import *
from ctypes.test import need_symbol
import _ctypes_test
class stdcall_dll(WinDLL):

    def __getattr__(self, name):
        if name[:2] == '__' and name[-2:] == '__':
            raise AttributeError(name)
        func = self._FuncPtr(('s_' + name, self))
        setattr(self, name, func)
        return func