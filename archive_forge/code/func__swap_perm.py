from .cusps import CuspPostDrillInfo
from .geometric_structure import compute_r13_planes_for_tet
from .tracing import compute_plane_intersection_param, Endpoint, GeodesicPiece
from .epsilons import compute_epsilon
from . import constants
from . import exceptions
from ..snap.t3mlite import simplex, Perm4, Tetrahedron # type: ignore
from ..matrix import matrix # type: ignore
from ..exceptions import InsufficientPrecisionError # type: ignore
from typing import Sequence, Optional, Union, Tuple, List, Dict, Mapping
def _swap_perm(i, j):
    result = [0, 1, 2, 3]
    result[i] = j
    result[j] = i
    return result