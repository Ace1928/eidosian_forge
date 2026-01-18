import unittest
from ctypes import *
from ctypes.test import need_symbol
import _ctypes_test
@need_symbol('WinDLL')
class stdcallCFunctions(CFunctions):
    _dll = stdcall_dll(_ctypes_test.__file__)