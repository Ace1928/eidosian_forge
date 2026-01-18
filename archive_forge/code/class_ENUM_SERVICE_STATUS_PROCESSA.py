from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
class ENUM_SERVICE_STATUS_PROCESSA(Structure):
    _fields_ = [('lpServiceName', LPSTR), ('lpDisplayName', LPSTR), ('ServiceStatusProcess', SERVICE_STATUS_PROCESS)]