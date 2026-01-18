from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
@staticmethod
def _exp_distance_edge(edge):
    """
        Given an edge, returns the exp of the (hyperbolic) distance of the
        two cusp neighborhoods at the ends of the edge measured along that
        edge.
        """
    tet, perm = next(edge.embeddings())
    face = 15 - (1 << perm[3])
    ptolemy_sqr = tet.horotriangles[1 << perm[0]].lengths[face] * tet.horotriangles[1 << perm[1]].lengths[face]
    return abs(1 / ptolemy_sqr)