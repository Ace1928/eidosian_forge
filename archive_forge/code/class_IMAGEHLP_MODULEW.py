from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
class IMAGEHLP_MODULEW(Structure):
    _fields_ = [('SizeOfStruct', DWORD), ('BaseOfImage', DWORD), ('ImageSize', DWORD), ('TimeDateStamp', DWORD), ('CheckSum', DWORD), ('NumSyms', DWORD), ('SymType', DWORD), ('ModuleName', WCHAR * 32), ('ImageName', WCHAR * 256), ('LoadedImageName', WCHAR * 256)]