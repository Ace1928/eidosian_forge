from OpenGL.error import _ErrorChecker, EGLError
from OpenGL import platform as _p
class EGLError(EGLError):

    @property
    def err(self):
        from OpenGL.EGL import debug
        return debug.eglErrorName(self.__dict__.get('err'))

    @err.setter
    def err(self, value):
        self.__dict__['err'] = value