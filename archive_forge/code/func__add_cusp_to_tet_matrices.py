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
def _add_cusp_to_tet_matrices(self):
    for tet in self.mcomplex.Tetrahedra:
        m = [(V, _compute_cusp_to_tet_and_inverse_matrices(tet, V, i)) for i, V in enumerate(t3m.ZeroSubsimplices)]
        tet.cusp_to_tet_matrices = {V: m1 for V, (m1, m2) in m}
        tet.tet_to_cusp_matrices = {V: m2 for V, (m1, m2) in m}