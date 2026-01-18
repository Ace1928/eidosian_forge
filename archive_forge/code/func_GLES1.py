import ctypes, ctypes.util
from OpenGL.platform import baseplatform, ctypesloader
@baseplatform.lazy_property
def GLES1(self):
    try:
        return ctypesloader.loadLibrary(ctypes.cdll, 'GLESv1_CM', mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return None