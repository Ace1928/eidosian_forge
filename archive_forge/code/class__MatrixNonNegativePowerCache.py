from .spatial_dict import SpatialDict, floor_as_integers
from .line import R13Line, R13LineWithMatrix
from .geodesic_info import LiftedTetrahedron
from ..hyperboloid import ( # type: ignore
from ..snap.t3mlite import Mcomplex # type: ignore
from ..matrix import matrix # type: ignore
class _MatrixNonNegativePowerCache:

    def __init__(self, m):
        self._m = m
        self._powers = [matrix.identity(ring=m.base_ring(), n=m.dimensions()[0])]

    def power(self, i):
        while not i < len(self._powers):
            self._powers.append(self._m * self._powers[-1])
        return self._powers[i]