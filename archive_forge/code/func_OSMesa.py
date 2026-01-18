import ctypes, ctypes.util
from OpenGL.platform import baseplatform, ctypesloader
from OpenGL.constant import Constant
from OpenGL.raw.osmesa import _types
@baseplatform.lazy_property
def OSMesa(self):
    return self.GL