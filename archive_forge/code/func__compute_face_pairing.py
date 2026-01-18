from snappy.snap import t3mlite as t3m
from snappy import Triangulation
from snappy.SnapPy import matrix, vector
from snappy.upper_halfspace import pgl2c_to_o13
from .hyperboloid_utilities import *
from .raytracing_data import *
def _compute_face_pairing(tet, F):
    tet_perm = _face_to_perm[F]
    m = tet.permutahedron_matrices[tet_perm.tuple()]
    other_tet_perm = tet.Gluing[F] * tet_perm
    other_tet = tet.Neighbor[F]
    other_m = other_tet.permutahedron_matrices[other_tet_perm.tuple()]
    return pgl2c_to_o13(_adjoint(other_m) * m)