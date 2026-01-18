from winappdbg.win32.defines import *
class MODULEINFO(Structure):
    _fields_ = [('lpBaseOfDll', LPVOID), ('SizeOfImage', DWORD), ('EntryPoint', LPVOID)]