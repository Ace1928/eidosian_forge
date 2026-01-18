from ..drilling import compute_geodesic_info
from ..drilling.geodesic_tube import GeodesicTube
from ..drilling.line import distance_r13_lines
from ..snap.t3mlite import simplex # type: ignore
def _get_pieces_covering_geodesic(self):
    if not self._pieces_covering_geodesic:
        self.geodesic_tube.add_pieces_for_radius(0)
        for piece in self.geodesic_tube.pieces:
            if piece.lower_bound > 0:
                break
            self._pieces_covering_geodesic.append(piece)
    return self._pieces_covering_geodesic