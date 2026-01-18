from OpenGL.arrays.arraydatatype import ArrayDatatype
from OpenGL.arrays.formathandler import FormatHandler
from OpenGL.raw.GL import _types 
from OpenGL import error
from OpenGL._bytes import bytes,unicode,as_8_bit
import ctypes,logging
from OpenGL._bytes import long, integer_types
import weakref
from OpenGL import acceleratesupport
class VBOOffsetHandler(VBOHandler):
    """Handles VBOOffset instances passed in as array data
        
        Registered on module import to provide support for VBOOffset instances 
        as sources for array data.
        """

    def dataPointer(self, instance):
        """Retrieve data-pointer from the instance's data

            returns instance' offset
            """
        return instance.offset

    def from_param(self, instance, typeCode=None):
        """Returns a c_void_p( instance.offset )"""
        return ctypes.c_void_p(instance.offset)