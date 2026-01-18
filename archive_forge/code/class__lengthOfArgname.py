from OpenGL.raw import GLE as raw
from OpenGL.raw.GLE import annotations as _simple
from OpenGL import wrapper, arrays
class _lengthOfArgname(object):
    """Calculates the length of a given argname over a divisor value"""

    def __init__(self, arrayName, divisor, arrayType=arrays.GLdoubleArray):
        self.arrayName = arrayName
        self.divisor = divisor
        self.arrayType = arrayType

    def finalise(self, wrapper):
        self.arrayIndex = wrapper.pyArgIndex(self.arrayName)

    def __call__(self, pyArgs, index, wrappedOperation):
        """Get the length of pyArgs[2], a glDoubleArray"""
        return self.arrayType.arraySize(pyArgs[self.arrayIndex]) // self.divisor