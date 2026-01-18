import ctypes
from OpenGL.platform import ctypesloader
from OpenGL._bytes import as_8_bit
import sys, logging
from OpenGL import _configflags
from OpenGL import logs, MODULE_ANNOTATIONS
class _DeprecatedFunctionPointer(_NullFunctionPointer):
    deprecated = True

    def __call__(self, *args, **named):
        from OpenGL import error
        raise error.NullFunctionError('Attempt to call a deprecated function %s while OpenGL in FORWARD_COMPATIBLE_ONLY mode.  Set OpenGL.FORWARD_COMPATIBLE_ONLY to False to use legacy entry points' % (self.__name__,))