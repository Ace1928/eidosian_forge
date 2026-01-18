import ctypes, ctypes.util
from OpenGL.platform import baseplatform, ctypesloader
class DarwinPlatform(baseplatform.BasePlatform):
    """Darwin (OSX) platform implementation"""
    DEFAULT_FUNCTION_TYPE = staticmethod(ctypes.CFUNCTYPE)
    EXTENSIONS_USE_BASE_FUNCTIONS = True

    @baseplatform.lazy_property
    def GL(self):
        try:
            return ctypesloader.loadLibrary(ctypes.cdll, 'OpenGL', mode=ctypes.RTLD_GLOBAL)
        except OSError as err:
            raise ImportError('Unable to load OpenGL library', *err.args)

    @baseplatform.lazy_property
    def GLU(self):
        return self.GL

    @baseplatform.lazy_property
    def CGL(self):
        return self.GL

    @baseplatform.lazy_property
    def GLUT(self):
        try:
            return ctypesloader.loadLibrary(ctypes.cdll, 'GLUT', mode=ctypes.RTLD_GLOBAL)
        except OSError:
            return None

    @baseplatform.lazy_property
    def GLE(self):
        return self.GLUT

    @baseplatform.lazy_property
    def GetCurrentContext(self):
        return self.CGL.CGLGetCurrentContext

    def getGLUTFontPointer(self, constant):
        """Platform specific function to retrieve a GLUT font pointer
        
        GLUTAPI void *glutBitmap9By15;
        #define GLUT_BITMAP_9_BY_15     (&glutBitmap9By15)
        
        Key here is that we want the addressof the pointer in the DLL,
        not the pointer in the DLL.  That is, our pointer is to the 
        pointer defined in the DLL, we don't want the *value* stored in
        that pointer.
        """
        name = [x.title() for x in constant.split('_')[1:]]
        internal = 'glut' + ''.join([x.title() for x in name])
        pointer = ctypes.c_void_p.in_dll(self.GLUT, internal)
        return ctypes.c_void_p(ctypes.addressof(pointer))

    @baseplatform.lazy_property
    def glGetError(self):
        return self.GL.glGetError