import re
import numpy
import cupy
import cupy._core._routines_manipulation as _manipulation
from cupy._core._dtype import get_dtype, _raise_if_invalid_cast
from cupy._core import internal
def _transpose_element(self, arg, iax, shape):
    iax = tuple((a if a < 0 else a - len(shape) for a in iax))
    tidc = tuple((i for i in range(-len(shape) + 0, 0) if i not in iax)) + iax
    return arg.transpose(tidc)