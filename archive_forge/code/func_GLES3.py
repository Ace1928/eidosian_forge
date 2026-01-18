import ctypes, ctypes.util
from OpenGL.platform import baseplatform, ctypesloader
@baseplatform.lazy_property
def GLES3(self):
    return self.GLES2