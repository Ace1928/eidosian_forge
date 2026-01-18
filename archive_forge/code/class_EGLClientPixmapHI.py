import ctypes
from OpenGL._opaque import opaque_pointer_cls as _opaque_pointer_cls
from OpenGL import platform as _p
from OpenGL import extensions 
from OpenGL._bytes import as_8_bit
class EGLClientPixmapHI(ctypes.Structure):
    _fields_ = [('pData', ctypes.c_voidp), ('iWidth', EGLint), ('iHeight', EGLint), ('iStride', EGLint)]