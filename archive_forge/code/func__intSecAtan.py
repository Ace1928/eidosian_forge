from fontTools.misc.arrayTools import calcBounds, sectRect, rectArea
from fontTools.misc.transform import Identity
import math
from collections import namedtuple
from math import sqrt, acos, cos, pi
@cython.cfunc
@cython.inline
@cython.returns(cython.double)
@cython.locals(x=cython.double)
def _intSecAtan(x):
    return x * math.sqrt(x ** 2 + 1) / 2 + math.asinh(x) / 2