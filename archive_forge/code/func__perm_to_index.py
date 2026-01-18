from .cusps import CuspPostDrillInfo
from .tracing import GeodesicPiece
from .peripheral_curves import install_peripheral_curves
from ..snap.t3mlite import Tetrahedron, Perm4, Mcomplex, simplex
from typing import Dict, Tuple, List, Sequence
def _perm_to_index(perm: Perm4) -> int:
    return _perm_tuple_to_index[perm.tuple()]