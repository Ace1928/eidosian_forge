from ...sage_helper import _within_sage
from ..upper_halfspace.finite_point import *
def _to_vec(self, z):
    v = self._matrix * vector([z.real(), z.imag()])
    for e in v:
        if not e.absolute_diameter() < 0.5:
            raise RuntimeError('Too large interval encountered when tiling. Increasing the precision should fix this.')
    return v