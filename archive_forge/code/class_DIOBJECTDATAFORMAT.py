import ctypes
from pyglet.libs.win32 import com
class DIOBJECTDATAFORMAT(ctypes.Structure):
    _fields_ = (('pguid', ctypes.POINTER(com.GUID)), ('dwOfs', DWORD), ('dwType', DWORD), ('dwFlags', DWORD))
    __slots__ = [n for n, t in _fields_]