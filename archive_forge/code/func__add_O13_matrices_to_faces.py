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
def _add_O13_matrices_to_faces(self):
    for tet in self.mcomplex.Tetrahedra:
        tet.O13_matrices = {F: _o13_matrix_for_face(tet, F) for F in t3m.TwoSubsimplices}