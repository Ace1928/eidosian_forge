from winappdbg.win32.defines import *
def GetCurrentProcess():
    _GetCurrentProcess = windll.kernel32.GetCurrentProcess
    _GetCurrentProcess.argtypes = []
    _GetCurrentProcess.restype = HANDLE
    return _GetCurrentProcess()