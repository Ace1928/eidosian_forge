from ctypes import *
from .base import FontException
import pyglet.lib
class FT_LibraryRec(Structure):
    _fields_ = [('dummy', c_int)]

    def __del__(self):
        global _library
        try:
            print('FT_LibraryRec.__del__')
            FT_Done_FreeType(byref(self))
            _library = None
        except:
            pass