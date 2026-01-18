from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
@staticmethod
def _cusp_area(cusp):
    area = 0
    for corner in cusp.Corners:
        subsimplex = corner.Subsimplex
        area += corner.Tetrahedron.horotriangles[subsimplex].area
    return area