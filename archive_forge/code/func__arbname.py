from OpenGL.arrays.arraydatatype import ArrayDatatype
from OpenGL.arrays.formathandler import FormatHandler
from OpenGL.raw.GL import _types 
from OpenGL import error
from OpenGL._bytes import bytes,unicode,as_8_bit
import ctypes,logging
from OpenGL._bytes import long, integer_types
import weakref
from OpenGL import acceleratesupport
def _arbname(self, name):
    return (name.startswith('gl') and name.endswith('ARB') or (name.startswith('GL_') and name.endswith('ARB'))) and name != 'glInitVertexBufferObjectARB'