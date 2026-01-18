import sys
import ctypes
from pyglet.util import debug_print
class IUnknown(Interface):
    _methods_ = [('QueryInterface', STDMETHOD(REFIID, ctypes.c_void_p)), ('AddRef', METHOD(ctypes.c_int)), ('Release', METHOD(ctypes.c_int))]