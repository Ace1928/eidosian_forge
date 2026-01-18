import ctypes, ctypes.util
from OpenGL.platform import baseplatform, ctypesloader
@baseplatform.lazy_property
def OpenGL(self):
    return self.GL