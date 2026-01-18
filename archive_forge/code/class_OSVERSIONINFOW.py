from winappdbg.win32.defines import *
class OSVERSIONINFOW(Structure):
    _fields_ = [('dwOSVersionInfoSize', DWORD), ('dwMajorVersion', DWORD), ('dwMinorVersion', DWORD), ('dwBuildNumber', DWORD), ('dwPlatformId', DWORD), ('szCSDVersion', WCHAR * 128)]