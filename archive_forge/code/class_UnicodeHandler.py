from OpenGL.raw.GL import _types 
from OpenGL.raw.GL.VERSION import GL_1_1
from OpenGL.arrays import formathandler
import ctypes
from OpenGL import _bytes, error
from OpenGL._configflags import ERROR_ON_COPY
class UnicodeHandler(StringHandler):
    HANDLED_TYPES = (_bytes.unicode,)

    @classmethod
    def from_param(cls, value, typeCode=None):
        converted = _bytes.as_8_bit(value)
        result = StringHandler.from_param(converted)
        if converted is not value:
            if ERROR_ON_COPY:
                raise error.CopyError('Unicode string passed, cannot copy with ERROR_ON_COPY set, please use 8-bit strings')
            result._temporary_array_ = converted
        return result

    def asArray(self, value, typeCode=None):
        value = _bytes.as_8_bit(value)
        return StringHandler.asArray(self, value, typeCode=typeCode)