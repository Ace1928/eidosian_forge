from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
class IMAGEHLP_MODULEW64(Structure):
    _fields_ = [('SizeOfStruct', DWORD), ('BaseOfImage', DWORD64), ('ImageSize', DWORD), ('TimeDateStamp', DWORD), ('CheckSum', DWORD), ('NumSyms', DWORD), ('SymType', DWORD), ('ModuleName', WCHAR * 32), ('ImageName', WCHAR * 256), ('LoadedImageName', WCHAR * 256), ('LoadedPdbName', WCHAR * 256), ('CVSig', DWORD), ('CVData', WCHAR * (MAX_PATH * 3)), ('PdbSig', DWORD), ('PdbSig70', GUID), ('PdbAge', DWORD), ('PdbUnmatched', BOOL), ('DbgUnmatched', BOOL), ('LineNumbers', BOOL), ('GlobalSymbols', BOOL), ('TypeInfo', BOOL), ('SourceIndexed', BOOL), ('Publics', BOOL)]