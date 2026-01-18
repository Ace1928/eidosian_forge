from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def GetGUIThreadInfo(idThread):
    _GetGUIThreadInfo = windll.user32.GetGUIThreadInfo
    _GetGUIThreadInfo.argtypes = [DWORD, LPGUITHREADINFO]
    _GetGUIThreadInfo.restype = bool
    _GetGUIThreadInfo.errcheck = RaiseIfZero
    gui = GUITHREADINFO()
    _GetGUIThreadInfo(idThread, byref(gui))
    return gui