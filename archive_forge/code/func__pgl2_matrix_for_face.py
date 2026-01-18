from snappy.snap import t3mlite as t3m
from snappy import Triangulation
from snappy.SnapPy import matrix, vector
from snappy.snap.mcomplex_base import *
from snappy.verify.cuspCrossSection import *
from ..upper_halfspace import pgl2c_to_o13, sl2c_inverse
from ..upper_halfspace.ideal_point import ideal_point_to_r13
from .hyperboloid_utilities import *
from .upper_halfspace_utilities import *
from .raytracing_data import *
from math import sqrt
def _pgl2_matrix_for_face(tet, F):
    gluing = tet.Gluing[F]
    other_tet = tet.Neighbor[F]
    verts = [tet.complex_vertices[V] for V in t3m.ZeroSubsimplices if V & F]
    other_verts = [other_tet.complex_vertices[gluing.image(V)] for V in t3m.ZeroSubsimplices if V & F]
    m1 = pgl2_matrix_taking_0_1_inf_to_given_points(*verts)
    m2 = pgl2_matrix_taking_0_1_inf_to_given_points(*other_verts)
    return m2 * sl2c_inverse(m1)