from winappdbg.win32.defines import *
from winappdbg.win32.peb_teb import *
class FILE_NAME_INFORMATION(Structure):
    _fields_ = [('FileNameLength', ULONG), ('FileName', WCHAR * 1)]