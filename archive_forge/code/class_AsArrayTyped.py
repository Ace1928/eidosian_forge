import OpenGL
import ctypes
from OpenGL import _configflags
from OpenGL import contextdata, error, converters
from OpenGL.arrays import arraydatatype
from OpenGL._bytes import bytes,unicode
import logging
from OpenGL import acceleratesupport
class AsArrayTyped(converters.PyConverter):
    """Given arrayName and arrayType, convert arrayName to array of type
        
        TODO: It should be possible to drop this if ERROR_ON_COPY,
        as array inputs always have to be the final objects in that 
        case.
        """
    argNames = ('arrayName', 'arrayType')
    indexLookups = (('arrayIndex', 'arrayName', 'pyArgIndex'),)

    def __init__(self, arrayName='pointer', arrayType=None):
        self.arrayName = arrayName
        self.arrayType = arrayType

    def __call__(self, arg, wrappedOperation, args):
        """Get the arg as an array of the appropriate type"""
        return self.arrayType.asArray(arg)