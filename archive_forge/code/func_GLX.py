import ctypes, ctypes.util
from OpenGL.platform import baseplatform, ctypesloader
@baseplatform.lazy_property
def GLX(self):
    try:
        return ctypesloader.loadLibrary(ctypes.cdll, 'GLX', mode=ctypes.RTLD_GLOBAL)
    except OSError as err:
        return self.GL