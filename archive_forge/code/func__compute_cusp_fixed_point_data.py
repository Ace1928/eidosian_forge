from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
def _compute_cusp_fixed_point_data(self, cusp):
    """
        Compute abs(z-1), z, p0, p1 for each horotriangle, vertex and edge
        as described in _compute_cusp_fixed_point.
        """
    for corner in cusp.Corners:
        tet0, vert0 = (corner.Tetrahedron, corner.Subsimplex)
        vertex_link = _face_edge_face_triples_for_vertex_link[vert0]
        for face0, edge0, other_face in vertex_link:
            tet1, face1, vert1 = CuspCrossSectionBase._glued_to(tet0, face0, vert0)
            edge1 = tet0.Gluing[face0].image(edge0)
            trig0 = tet0.horotriangles[vert0]
            l0 = trig0.lengths[face0]
            p0 = trig0.vertex_positions[edge0]
            trig1 = tet1.horotriangles[vert1]
            l1 = trig1.lengths[face1]
            p1 = trig1.vertex_positions[edge1]
            z = -l1 / l0
            yield (abs(z - 1), z, p0, p1)