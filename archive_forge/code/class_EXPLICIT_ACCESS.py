import ctypes
from os_win.utils.winapi import wintypes
class EXPLICIT_ACCESS(ctypes.Structure):
    _fields_ = [('grfAccessPermissions', wintypes.DWORD), ('grfAccessMode', wintypes.INT), ('grfInheritance', wintypes.DWORD), ('Trustee', TRUSTEE)]