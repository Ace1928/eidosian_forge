from snappy.snap import t3mlite as t3m
from snappy import Triangulation
from snappy.SnapPy import matrix, vector
from snappy.upper_halfspace import pgl2c_to_o13
from .hyperboloid_utilities import *
from .raytracing_data import *
def _compute_edge_ends(tet, perm):
    m = tet.permutahedron_matrices[perm]
    return [pgl2c_to_o13(_adjoint(m)) * c for c in cs]