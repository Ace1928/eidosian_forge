import ctypes
import OpenGL
from OpenGL.raw.GL import _types
from OpenGL import plugins
from OpenGL.arrays import formathandler, _arrayconstants as GL_1_1
from OpenGL import logs
from OpenGL import acceleratesupport
class ArrayDatatype(object):
    """Mix-in for array datatype classes

        The ArrayDatatype marker essentially is used to mark a particular argument
        as having an "array" type, which means that it is eligible for handling
        via the arrays sub-package and its registered handlers.
        """
    typeConstant = None
    handler = GLOBAL_REGISTRY
    getHandler = GLOBAL_REGISTRY.__call__
    returnHandler = GLOBAL_REGISTRY.get_output_handler
    isAccelerated = False

    @classmethod
    def getRegistry(cls):
        """Get our handler registry"""
        return cls.handler

    def from_param(cls, value, typeConstant=None):
        """Given a value in a known data-pointer type, convert to a ctypes pointer"""
        return cls.getHandler(value).from_param(value, cls.typeConstant)
    from_param = classmethod(logs.logOnFail(from_param, _log))

    def dataPointer(cls, value):
        """Given a value in a known data-pointer type, return long for pointer"""
        try:
            return cls.getHandler(value).dataPointer(value)
        except Exception:
            _log.warning('Failure in dataPointer for %s instance %s', type(value), value)
            raise
    dataPointer = classmethod(logs.logOnFail(dataPointer, _log))

    def voidDataPointer(cls, value):
        """Given value in a known data-pointer type, return void_p for pointer"""
        pointer = cls.dataPointer(value)
        try:
            return ctypes.c_void_p(pointer)
        except TypeError:
            return pointer
    voidDataPointer = classmethod(logs.logOnFail(voidDataPointer, _log))

    def typedPointer(cls, value):
        """Return a pointer-to-base-type pointer for given value"""
        return ctypes.cast(cls.dataPointer(value), ctypes.POINTER(cls.baseType))
    typedPointer = classmethod(typedPointer)

    def asArray(cls, value, typeCode=None):
        """Given a value, convert to preferred array representation"""
        return cls.getHandler(value).asArray(value, typeCode or cls.typeConstant)
    asArray = classmethod(logs.logOnFail(asArray, _log))

    def arrayToGLType(cls, value):
        """Given a data-value, guess the OpenGL type of the corresponding pointer

            Note: this is not currently used in PyOpenGL and may be removed
            eventually.
            """
        return cls.getHandler(value).arrayToGLType(value)
    arrayToGLType = classmethod(logs.logOnFail(arrayToGLType, _log))

    def arraySize(cls, value, typeCode=None):
        """Given a data-value, calculate dimensions for the array (number-of-units)"""
        return cls.getHandler(value).arraySize(value, typeCode or cls.typeConstant)
    arraySize = classmethod(logs.logOnFail(arraySize, _log))

    def unitSize(cls, value, typeCode=None):
        """Determine unit size of an array (if possible)

            Uses our local type if defined, otherwise asks the handler to guess...
            """
        return cls.getHandler(value).unitSize(value, typeCode or cls.typeConstant)
    unitSize = classmethod(logs.logOnFail(unitSize, _log))

    def zeros(cls, dims, typeCode=None):
        """Allocate a return array of the given dimensions filled with zeros"""
        return cls.returnHandler().zeros(dims, typeCode or cls.typeConstant)
    zeros = classmethod(logs.logOnFail(zeros, _log))

    def dimensions(cls, value):
        """Given a data-value, get the dimensions (assumes full structure info)"""
        return cls.getHandler(value).dimensions(value)
    dimensions = classmethod(logs.logOnFail(dimensions, _log))

    def arrayByteCount(cls, value):
        """Given a data-value, try to determine number of bytes it's final form occupies

            For most data-types this is arraySize() * atomic-unit-size
            """
        return cls.getHandler(value).arrayByteCount(value)
    arrayByteCount = classmethod(logs.logOnFail(arrayByteCount, _log))