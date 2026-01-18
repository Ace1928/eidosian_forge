from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
@staticmethod
def _glued_to(tetrahedron, face, vertex):
    """
        Returns (other tet, other face, other vertex).
        """
    gluing = tetrahedron.Gluing[face]
    return (tetrahedron.Neighbor[face], gluing.image(face), gluing.image(vertex))