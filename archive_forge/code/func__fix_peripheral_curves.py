from .cusps import CuspPostDrillInfo
from .tracing import GeodesicPiece
from .peripheral_curves import install_peripheral_curves
from ..snap.t3mlite import Tetrahedron, Perm4, Mcomplex, simplex
from typing import Dict, Tuple, List, Sequence
def _fix_peripheral_curves(subtet):
    """
    Traverse the six new cusp triangles shown in one
    of the figures in crush_geodesic_pieces.
    """
    for i in range(6):
        subtet.needs_peripheral_curves_fixed = False
        if i % 2 == 0:
            face0, face1 = (simplex.F1, simplex.F2)
        else:
            face0, face1 = (simplex.F2, simplex.F1)
        neighbor = subtet.Neighbor[face1]
        for ml in range(2):
            for sheet in range(2):
                tri = subtet.PeripheralCurves[ml][sheet][simplex.V0]
                p = tri[face0] + tri[simplex.F3]
                tri[face1] = -p
                neighbor.PeripheralCurves[ml][1 - sheet][simplex.V0][face1] = p
        subtet = neighbor