import OpenGL
import ctypes
from OpenGL import _configflags
from OpenGL import contextdata, error, converters
from OpenGL.arrays import arraydatatype
from OpenGL._bytes import bytes,unicode
import logging
from OpenGL import acceleratesupport
class AsArrayTypedSize(converters.CConverter):
    """Given arrayName and arrayType, determine size of arrayName
        """
    argNames = ('arrayName', 'arrayType')
    indexLookups = (('arrayIndex', 'arrayName', 'pyArgIndex'),)

    def __init__(self, arrayName='pointer', arrayType=None):
        self.arrayName = arrayName
        self.arrayType = arrayType

    def __call__(self, pyArgs, index, wrappedOperation):
        """Get the arg as an array of the appropriate type"""
        return self.arrayType.arraySize(pyArgs[self.arrayIndex])