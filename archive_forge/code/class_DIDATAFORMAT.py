import ctypes
from pyglet.libs.win32 import com
class DIDATAFORMAT(ctypes.Structure):
    _fields_ = (('dwSize', DWORD), ('dwObjSize', DWORD), ('dwFlags', DWORD), ('dwDataSize', DWORD), ('dwNumObjs', DWORD), ('rgodf', LPDIOBJECTDATAFORMAT))
    __slots__ = [n for n, t in _fields_]