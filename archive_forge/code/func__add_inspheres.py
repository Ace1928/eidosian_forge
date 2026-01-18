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
def _add_inspheres(self):
    for tet in self.mcomplex.Tetrahedra:
        tet.inradius = tet.R13_planes[t3m.F0][0].arcsinh()
        tmp = tet.inradius * self.insphere_scale
        tet.cosh_sqr_inradius = tmp.cosh() ** 2