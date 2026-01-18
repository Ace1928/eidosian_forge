from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
class ENUM_SERVICE_STATUS_PROCESSW(Structure):
    _fields_ = [('lpServiceName', LPWSTR), ('lpDisplayName', LPWSTR), ('ServiceStatusProcess', SERVICE_STATUS_PROCESS)]