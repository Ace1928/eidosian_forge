import ctypes
from pyglet.libs.win32 import com
class DIPROPDWORD(ctypes.Structure):
    _fields_ = (('diph', DIPROPHEADER), ('dwData', DWORD))