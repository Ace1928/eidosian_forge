from ...sage_helper import _within_sage
from ...math_basics import correct_max
from ...snap.kernel_structures import *
from ...snap.fundamental_polyhedron import *
from ...snap.mcomplex_base import *
from ...snap.t3mlite import simplex
from ...snap import t3mlite as t3m
from ...exceptions import InsufficientPrecisionError
from ..cuspCrossSection import ComplexCuspCrossSection
from ..upper_halfspace.ideal_point import *
from ..interval_tree import *
from .cusp_translate_engine import *
import heapq
def _horo_intersection_data(self, vertex, is_at_infinity):
    corner = vertex.Corners[0]
    v0 = corner.Subsimplex
    v1, v2, v3 = simplex.VerticesOfFaceCounterclockwise[simplex.comp(v0)]
    vertices = [v0, v1, v2]
    face = v0 | v1 | v2
    tet = corner.Tetrahedron
    if is_at_infinity:
        idealPoints = [None, self._ideal_point(tet, v1), self._ideal_point(tet, v2)]
    else:
        idealPoints = [self._ideal_point(tet, v) for v in vertices]
    cusp_length = tet.horotriangles[v0].get_real_lengths()[face]
    return (idealPoints, cusp_length)