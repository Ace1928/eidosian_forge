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
def _add_complex_vertices(self):
    for tet in self.mcomplex.Tetrahedra:
        tet.complex_vertices = {v: vert for v, vert in zip(t3m.ZeroSubsimplices, symmetric_vertices_for_tetrahedron(tet.ShapeParameters[t3m.E01]))}