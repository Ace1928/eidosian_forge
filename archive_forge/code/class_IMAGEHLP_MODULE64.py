from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
class IMAGEHLP_MODULE64(Structure):
    _fields_ = [('SizeOfStruct', DWORD), ('BaseOfImage', DWORD64), ('ImageSize', DWORD), ('TimeDateStamp', DWORD), ('CheckSum', DWORD), ('NumSyms', DWORD), ('SymType', DWORD), ('ModuleName', CHAR * 32), ('ImageName', CHAR * 256), ('LoadedImageName', CHAR * 256), ('LoadedPdbName', CHAR * 256), ('CVSig', DWORD), ('CVData', CHAR * (MAX_PATH * 3)), ('PdbSig', DWORD), ('PdbSig70', GUID), ('PdbAge', DWORD), ('PdbUnmatched', BOOL), ('DbgUnmatched', BOOL), ('LineNumbers', BOOL), ('GlobalSymbols', BOOL), ('TypeInfo', BOOL), ('SourceIndexed', BOOL), ('Publics', BOOL)]