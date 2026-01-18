from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
def _add_one_cusp_vertex_positions(self, cusp):
    """
        Procedure is similar to _add_one_cusp_cross_section
        """
    corner0 = cusp.Corners[0]
    tet0, vert0 = (corner0.Tetrahedron, corner0.Subsimplex)
    zero = tet0.ShapeParameters[t3m.simplex.E01].parent()(0)
    tet0.horotriangles[vert0].add_vertex_positions(vert0, _pick_an_edge_for_vertex[vert0], zero)
    active = [(tet0, vert0)]
    visited = set()
    while active:
        tet0, vert0 = active.pop()
        for face0 in t3m.simplex.FacesAroundVertexCounterclockwise[vert0]:
            tet1, face1, vert1 = CuspCrossSectionBase._glued_to(tet0, face0, vert0)
            if not (tet1.Index, vert1) in visited:
                edge0 = _pick_an_edge_for_vertex_and_face[vert0, face0]
                edge1 = tet0.Gluing[face0].image(edge0)
                tet1.horotriangles[vert1].add_vertex_positions(vert1, edge1, tet0.horotriangles[vert0].vertex_positions[edge0])
                active.append((tet1, vert1))
                visited.add((tet1.Index, vert1))