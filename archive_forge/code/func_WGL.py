import ctypes
import platform
from OpenGL.platform import ctypesloader, baseplatform
import sys
@baseplatform.lazy_property
def WGL(self):
    return self.OpenGL