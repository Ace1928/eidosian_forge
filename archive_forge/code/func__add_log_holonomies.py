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
def _add_log_holonomies(self):
    shapes = [tet.ShapeParameters[e].log() for tet in self.mcomplex.Tetrahedra for e in [t3m.E01, t3m.E02, t3m.E03]]
    for cusp, cusp_info in zip(self.mcomplex.Vertices, self.snappy_manifold.cusp_info()):
        self._add_log_holonomies_to_cusp(cusp, shapes)