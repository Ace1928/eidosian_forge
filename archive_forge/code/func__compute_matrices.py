from snappy.snap import t3mlite as t3m
from snappy import Triangulation
from snappy.SnapPy import matrix, vector
from snappy.upper_halfspace import pgl2c_to_o13
from .hyperboloid_utilities import *
from .raytracing_data import *
def _compute_matrices(self, hyperbolic_structure):
    for tet in self.mcomplex.Tetrahedra:
        tet.permutahedron_matrices = _matrices_for_tet(hyperbolic_structure, tet.Index)