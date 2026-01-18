from OpenGL.raw import GLE as raw
from OpenGL.raw.GLE import annotations as _simple
from OpenGL import wrapper, arrays
def _baseWrap(base, lengthName='ncp', contourName='contour', divisor=2):
    """Do the basic wrapping operation for a GLE function"""
    return wrapper.wrapper(base).setPyConverter(lengthName).setCConverter(lengthName, _lengthOfArgname(contourName, divisor, arrays.GLdoubleArray))