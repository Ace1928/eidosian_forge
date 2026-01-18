from .geodesic_tube_info import GeodesicTubeInfo
from .upper_halfspace_utilities import *
from ..drilling.geometric_structure import add_r13_geometry
from ..drilling.geodesic_tube import add_structures_necessary_for_tube
from ..snap.t3mlite import Mcomplex, simplex
from ..upper_halfspace import pgl2c_to_o13, sl2c_inverse
def _hsv2rgb_helper(hue, saturation, value, x):
    p = abs((hue + x / 3.0) % 1.0 * 6.0 - 3.0)
    c = min(max(p - 1.0, 0.0), 1.0)
    return value * (1.0 + saturation * (c - 1.0))