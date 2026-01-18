from .geodesic_info import GeodesicInfo
from .line import R13LineWithMatrix, distance_r13_lines
from . import constants
from . import epsilons
from . import exceptions
from ..snap.t3mlite import simplex, Tetrahedron, Mcomplex # type: ignore
from ..hyperboloid import r13_dot # type: ignore
from ..exceptions import InsufficientPrecisionError # type: ignore
from typing import Sequence, Optional, List
class GeodesicPieceTracker:

    def __init__(self, geodesic_piece):
        self.set_geodesic_piece(geodesic_piece)

    def set_geodesic_piece(self, geodesic_piece):
        self.geodesic_piece = geodesic_piece
        geodesic_piece.tracker = self