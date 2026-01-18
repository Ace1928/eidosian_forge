from .cusps import CuspPostDrillInfo
from .tracing import GeodesicPiece
from .peripheral_curves import install_peripheral_curves
from ..snap.t3mlite import Tetrahedron, Perm4, Mcomplex, simplex
from typing import Dict, Tuple, List, Sequence

    Fix peripheral curves for all subtetrahedra that require it, see
    crush_geodesic_pieces where the needs_peripheral_curves_fixed flag
    was raised for details.
    