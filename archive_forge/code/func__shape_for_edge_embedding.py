from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
@staticmethod
def _shape_for_edge_embedding(tet, perm):
    """
        Given an edge embedding, find the shape assignment for it.
        If the edge embedding flips orientation, apply conjugate inverse.
        """
    subsimplex = perm.image(3)
    if perm.sign():
        return 1 / tet.ShapeParameters[subsimplex].conjugate()
    else:
        return tet.ShapeParameters[subsimplex]