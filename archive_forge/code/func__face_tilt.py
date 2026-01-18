from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
@staticmethod
def _face_tilt(face):
    """
        Tilt of a face in the trinagulation: this is the sum of
        the two tilts of the two faces of the two tetrahedra that are
        glued. The argument is a t3m.simplex.Face.
        """
    return sum([RealCuspCrossSection._tet_tilt(corner.Tetrahedron, corner.Subsimplex) for corner in face.Corners])