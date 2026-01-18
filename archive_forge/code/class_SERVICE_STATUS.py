from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
class SERVICE_STATUS(Structure):
    _fields_ = [('dwServiceType', DWORD), ('dwCurrentState', DWORD), ('dwControlsAccepted', DWORD), ('dwWin32ExitCode', DWORD), ('dwServiceSpecificExitCode', DWORD), ('dwCheckPoint', DWORD), ('dwWaitHint', DWORD)]