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
def _add_log_holonomies_to_cusp(self, cusp, shapes):
    i = cusp.Index
    if cusp.is_complete:
        m_param, l_param = cusp.Translations
    else:
        m_param, l_param = [sum((shape * expo for shape, expo in zip(shapes, self.peripheral_gluing_equations[2 * i + j]))) for j in range(2)]
    a, c = (m_param.real(), m_param.imag())
    b, d = (l_param.real(), l_param.imag())
    det = a * d - b * c
    cusp.mat_log = matrix([[d, -b], [-c, a]]) / det
    if cusp.is_complete:
        cusp.margulisTubeRadiusParam = 0.0
    else:
        slope = 2 * self.areas[i] / abs(det)
        x = (slope ** 2 / (slope ** 2 + 1)).sqrt()
        y = (1 / (slope ** 2 + 1)).sqrt()
        rSqr = 1 + (x ** 2 + (1 - y) ** 2) / (2 * y)
        cusp.margulisTubeRadiusParam = 0.25 * (1.0 + rSqr)