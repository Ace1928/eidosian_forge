from .spatial_dict import SpatialDict, floor_as_integers
from .line import R13Line, R13LineWithMatrix
from .geodesic_info import LiftedTetrahedron
from ..hyperboloid import ( # type: ignore
from ..snap.t3mlite import Mcomplex # type: ignore
from ..matrix import matrix # type: ignore
class _O13MatrixPowerCache:

    def __init__(self, m):
        self._positive_cache = _MatrixNonNegativePowerCache(m)
        self._negative_cache = _MatrixNonNegativePowerCache(o13_inverse(m))

    def power(self, i):
        if i >= 0:
            return self._positive_cache.power(i)
        else:
            return self._negative_cache.power(-i)