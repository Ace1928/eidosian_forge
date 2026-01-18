from winappdbg.win32.defines import *
from winappdbg.win32.peb_teb import *
def CsrGetProcessId():
    _CsrGetProcessId = windll.ntdll.CsrGetProcessId
    _CsrGetProcessId.argtypes = []
    _CsrGetProcessId.restype = DWORD
    return _CsrGetProcessId()