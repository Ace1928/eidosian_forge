import OpenGL
import ctypes
from OpenGL import _configflags
from OpenGL import contextdata, error, converters
from OpenGL.arrays import arraydatatype
from OpenGL._bytes import bytes,unicode
import logging
from OpenGL import acceleratesupport
class AsArrayOfType(converters.PyConverter):
    """Given arrayName and typeName coerce arrayName to array of type typeName
        
        TODO: It should be possible to drop this if ERROR_ON_COPY,
        as array inputs always have to be the final objects in that 
        case.
        """
    argNames = ('arrayName', 'typeName')
    indexLookups = (('arrayIndex', 'arrayName', 'pyArgIndex'), ('typeIndex', 'typeName', 'pyArgIndex'))

    def __init__(self, arrayName='pointer', typeName='type'):
        self.arrayName = arrayName
        self.typeName = typeName

    def __call__(self, arg, wrappedOperation, args):
        """Get the arg as an array of the appropriate type"""
        type = args[self.typeIndex]
        arrayType = arraydatatype.GL_CONSTANT_TO_ARRAY_TYPE[type]
        return arrayType.asArray(arg)