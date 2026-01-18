from OpenGL.raw.GL import _types 
from OpenGL.raw.GL.VERSION import GL_1_1
from OpenGL.arrays import formathandler
import ctypes
from OpenGL import _bytes, error
from OpenGL._configflags import ERROR_ON_COPY
class StringHandler(formathandler.FormatHandler):
    """String-specific data-type handler for OpenGL"""
    HANDLED_TYPES = (_bytes.bytes,)

    @classmethod
    def from_param(cls, value, typeCode=None):
        return ctypes.c_void_p(dataPointer(value))
    dataPointer = staticmethod(dataPointer)

    def zeros(self, dims, typeCode=None):
        """Currently don't allow strings as output types!"""
        raise NotImplemented("Don't currently support strings as output arrays")

    def ones(self, dims, typeCode=None):
        """Currently don't allow strings as output types!"""
        raise NotImplemented("Don't currently support strings as output arrays")

    def arrayToGLType(self, value):
        """Given a value, guess OpenGL type of the corresponding pointer"""
        raise NotImplemented("Can't guess data-type from a string-type argument")

    def arraySize(self, value, typeCode=None):
        """Given a data-value, calculate ravelled size for the array"""
        byteCount = BYTE_SIZES[typeCode]
        return len(value) // byteCount

    def arrayByteCount(self, value, typeCode=None):
        """Given a data-value, calculate number of bytes required to represent"""
        return len(value)

    def asArray(self, value, typeCode=None):
        """Convert given value to an array value of given typeCode"""
        if isinstance(value, bytes):
            return value
        elif hasattr(value, 'tostring'):
            return value.tostring()
        elif hasattr(value, 'raw'):
            return value.raw
        raise TypeError('String handler got non-string object: %r' % type(value))

    def dimensions(self, value, typeCode=None):
        """Determine dimensions of the passed array value (if possible)"""
        raise TypeError('Cannot calculate dimensions for a String data-type')