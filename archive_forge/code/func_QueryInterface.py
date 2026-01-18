import sys
import ctypes
from pyglet.util import debug_print
def QueryInterface(self, iid_ptr, res_ptr):
    ctypes.cast(res_ptr, ctypes.POINTER(ctypes.c_void_p))[0] = 0
    return E_NOINTERFACE