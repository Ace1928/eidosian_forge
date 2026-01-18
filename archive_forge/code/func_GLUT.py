import ctypes, ctypes.util
from OpenGL.platform import baseplatform, ctypesloader
@baseplatform.lazy_property
def GLUT(self):
    try:
        return ctypesloader.loadLibrary(ctypes.cdll, 'GLUT', mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return None