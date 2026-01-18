from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymSetParentWindow(hwnd):
    _SymSetParentWindow = windll.dbghelp.SymSetParentWindow
    _SymSetParentWindow.argtypes = [HWND]
    _SymSetParentWindow.restype = bool
    _SymSetParentWindow.errcheck = RaiseIfZero
    _SymSetParentWindow(hwnd)