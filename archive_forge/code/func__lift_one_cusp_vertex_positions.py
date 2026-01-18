from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
def _lift_one_cusp_vertex_positions(self, cusp):
    corner0 = cusp.Corners[0]
    tet0, vert0 = (corner0.Tetrahedron, corner0.Subsimplex)
    trig0 = tet0.horotriangles[vert0]
    edge0 = _pick_an_edge_for_vertex[vert0]
    if cusp.is_complete:
        for corner in cusp.Corners:
            tet0, vert0 = (corner.Tetrahedron, corner.Subsimplex)
            tet0.horotriangles[vert0].lifted_vertex_positions = {vert0 | vert1: None for vert1 in t3m.ZeroSubsimplices if vert0 != vert1}
        return
    trig0.lift_vertex_positions(log(trig0.vertex_positions[edge0]))
    active = [(tet0, vert0)]
    visited = set()
    while active:
        tet0, vert0 = active.pop()
        for face0 in t3m.simplex.FacesAroundVertexCounterclockwise[vert0]:
            tet1, face1, vert1 = CuspCrossSectionBase._glued_to(tet0, face0, vert0)
            if not (tet1.Index, vert1) in visited:
                edge0 = _pick_an_edge_for_vertex_and_face[vert0, face0]
                tet1.horotriangles[vert1].lift_vertex_positions(tet0.horotriangles[vert0].lifted_vertex_positions[edge0])
                active.append((tet1, vert1))
                visited.add((tet1.Index, vert1))