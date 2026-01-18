from OpenGL.raw.GL import _types
from OpenGL.raw.GL.VERSION import GL_1_1
from OpenGL.arrays import formathandler
import ctypes
from OpenGL._bytes import long, integer_types
class NumberHandler(formathandler.FormatHandler):
    """Allows the user to pass a bald Python float,int, etceteras as an array-of-1"""
    HANDLED_TYPES = integer_types + (float, _types.GLdouble, _types.GLfloat, _types.GLint, _types.GLshort, _types.GLuint, _types.GLulong, _types.GLushort, _types.GLclampf, _types.GLclampd)

    def from_param(self, value, typeCode=None):
        """If it's a ctypes value, pass on, otherwise do asArray"""
        try:
            return ctypes.byref(value)
        except TypeError as err:
            err.args += (' If you have ERROR_ON_COPY enabled, remember to pass in an array to array-requiring functions.',)
            raise
    dataPointer = from_param

    def zeros(self, dims, typeCode=None):
        """Currently don't allow Number as output types!"""
        raise NotImplemented('Number data-type not allowed as an output array format')

    def ones(self, dims, typeCode=None):
        """Currently don't allow Number as output types!"""
        raise NotImplemented('Number data-type not allowed as an output array format')

    def arrayToGLType(self, value):
        """Given a value, guess OpenGL type of the corresponding pointer"""
        if value.__class__ in TARGET_TYPES:
            return TARGET_TYPES[value.__class__]
        else:
            guess = DEFAULT_TYPES.get(value.__class__)
            if guess is not None:
                return guess[1]
            raise TypeError("Can't guess array data-type for %r types" % type(value))

    def arraySize(self, value, typeCode=None):
        """Given a data-value, calculate ravelled size for the array"""
        return 1

    def asArray(self, value, typeCode=None):
        """Convert given value to an array value of given typeCode"""
        if value.__class__ in TARGET_TYPES:
            return value
        targetType = CONSTANT_TO_TYPE.get(typeCode)
        if targetType is not None:
            return targetType(value)
        raise TypeError("Don't know how to convert %r to an array type" % (typeCode,))

    def unitSize(self, value, typeCode=None):
        """Determine unit size of an array (if possible)"""
        return 1

    def registerEquivalent(self, typ, base):
        """Register a sub-class for handling as the base-type"""
        global TARGET_TYPE_TUPLE
        for source in (DEFAULT_TYPES, TARGET_TYPES, BYTE_SIZES):
            if base in source:
                source[typ] = source[base]
        if base in TARGET_TYPES:
            TARGET_TYPE_TUPLE = TARGET_TYPE_TUPLE + (base,)