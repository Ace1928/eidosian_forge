import ctypes
from pyglet.libs.win32 import com
class DIDEVICEOBJECTDATA(ctypes.Structure):
    _fields_ = (('dwOfs', DWORD), ('dwData', DWORD), ('dwTimeStamp', DWORD), ('dwSequence', DWORD), ('uAppData', ctypes.POINTER(UINT)))