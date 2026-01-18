from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
@staticmethod
def _scale_cusp(cusp, scale):
    for corner in cusp.Corners:
        subsimplex = corner.Subsimplex
        corner.Tetrahedron.horotriangles[subsimplex].rescale(scale)