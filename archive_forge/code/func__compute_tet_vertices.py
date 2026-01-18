from snappy.snap import t3mlite as t3m
from snappy import Triangulation
from snappy.SnapPy import matrix, vector
from snappy.upper_halfspace import pgl2c_to_o13
from .hyperboloid_utilities import *
from .raytracing_data import *
def _compute_tet_vertices(self):
    c = vector(self.RF, [1, 0, 0, 0])

    def _compute_vertex(tet, perm):
        m = tet.permutahedron_matrices[perm]
        return pgl2c_to_o13(_adjoint(m)) * c
    for tet in self.mcomplex.Tetrahedra:
        tet.R13_vertices = {t3m.V0: _compute_vertex(tet, (0, 1, 3, 2)), t3m.V1: _compute_vertex(tet, (1, 0, 2, 3)), t3m.V2: _compute_vertex(tet, (2, 0, 3, 1)), t3m.V3: _compute_vertex(tet, (3, 0, 1, 2))}