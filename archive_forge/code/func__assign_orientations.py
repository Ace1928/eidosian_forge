from .cusps import CuspPostDrillInfo
from .tracing import GeodesicPiece
from .peripheral_curves import install_peripheral_curves
from ..snap.t3mlite import Tetrahedron, Perm4, Mcomplex, simplex
from typing import Dict, Tuple, List, Sequence
def _assign_orientations(subtetrahedra):
    for j in range(len(subtetrahedra) // 24):
        for i, perm in enumerate(Perm4.S4()):
            subtet_index = 24 * j + i
            subtet = subtetrahedra[subtet_index]
            if subtet:
                subtet.orientation = perm.sign()