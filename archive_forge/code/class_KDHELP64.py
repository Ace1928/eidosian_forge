from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
class KDHELP64(Structure):
    _fields_ = [('Thread', DWORD64), ('ThCallbackStack', DWORD), ('ThCallbackBStore', DWORD), ('NextCallback', DWORD), ('FramePointer', DWORD), ('KiCallUserMode', DWORD64), ('KeUserCallbackDispatcher', DWORD64), ('SystemRangeStart', DWORD64), ('KiUserExceptionDispatcher', DWORD64), ('StackBase', DWORD64), ('StackLimit', DWORD64), ('Reserved', DWORD64 * 5)]