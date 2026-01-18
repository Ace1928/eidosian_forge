from OpenGL.raw.GL.VERSION import GL_1_1,GL_1_2, GL_3_0
from OpenGL import images, arrays, wrapper
from OpenGL.arrays import arraydatatype
from OpenGL._bytes import bytes,integer_types
from OpenGL.raw.GL import _types
import ctypes
def _setDataSize(baseFunction, argument='imageSize'):
    """Set the data-size value to come from the data field"""
    converter = CompressedImageConverter()
    return asWrapper(baseFunction).setPyConverter(argument).setCConverter(argument, converter)