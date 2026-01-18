from snappy.snap import t3mlite as t3m
from snappy import Triangulation
from snappy.SnapPy import matrix, vector
from snappy.upper_halfspace import pgl2c_to_o13
from .hyperboloid_utilities import *
from .raytracing_data import *
def _compute_face_pairings(self):
    for tet in self.mcomplex.Tetrahedra:
        tet.O13_matrices = {F: _compute_face_pairing(tet, F) for F in t3m.TwoSubsimplices}