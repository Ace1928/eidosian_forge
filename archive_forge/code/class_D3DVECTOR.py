import ctypes
from pyglet.libs.win32 import com
class D3DVECTOR(ctypes.Structure):
    _fields_ = [('x', ctypes.c_float), ('y', ctypes.c_float), ('z', ctypes.c_float)]