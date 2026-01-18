from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
def _debug_show_lifted_horotriangles(self, cusp=0):
    from sage.all import line, real, imag
    self.add_vertex_positions_to_horotriangles()
    return sum([line([(real(z0), imag(z0)), (real(z1), imag(z1))]) for tet in self.mcomplex.Tetrahedra for V, h in tet.horotriangles.items() for z0 in h.lifted_vertex_positions.values() for z1 in h.lifted_vertex_positions.values() if tet.Class[V].Index == cusp])