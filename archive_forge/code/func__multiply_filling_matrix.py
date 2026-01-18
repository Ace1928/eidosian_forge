from .geodesic_info import GeodesicInfo
from .geometric_structure import Filling, FillingMatrix
from ..snap.t3mlite import Mcomplex, simplex
from typing import Tuple, Optional, Sequence
def _multiply_filling_matrix(m: FillingMatrix, s: int) -> FillingMatrix:
    return ((s * m[0][0], s * m[0][1]), (s * m[1][0], s * m[1][1]))