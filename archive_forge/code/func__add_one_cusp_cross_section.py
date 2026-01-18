from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
def _add_one_cusp_cross_section(self, cusp, one_cocycle):
    """
        Build a cusp cross section as described in Section 3.6 of the paper

        Asymmetric hyperbolic L-spaces, Heegaard genus, and Dehn filling
        Nathan M. Dunfield, Neil R. Hoffman, Joan E. Licata
        http://arxiv.org/abs/1407.7827
        """
    corner0 = cusp.Corners[0]
    tet0, vert0 = (corner0.Tetrahedron, corner0.Subsimplex)
    face0 = t3m.simplex.FacesAroundVertexCounterclockwise[vert0][0]
    tet0.horotriangles[vert0] = self.HoroTriangle(tet0, vert0, face0, 1)
    active = [(tet0, vert0)]
    while active:
        tet0, vert0 = active.pop()
        for face0 in t3m.simplex.FacesAroundVertexCounterclockwise[vert0]:
            tet1, face1, vert1 = CuspCrossSectionBase._glued_to(tet0, face0, vert0)
            if tet1.horotriangles[vert1] is None:
                known_side = self.HoroTriangle.direction_sign() * tet0.horotriangles[vert0].lengths[face0]
                if one_cocycle:
                    known_side *= one_cocycle[tet0.Index, face0, vert0]
                tet1.horotriangles[vert1] = self.HoroTriangle(tet1, vert1, face1, known_side)
                active.append((tet1, vert1))