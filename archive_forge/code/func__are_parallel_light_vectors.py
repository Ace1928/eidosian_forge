from ..drilling import compute_geodesic_info
from ..drilling.geodesic_tube import GeodesicTube
from ..drilling.line import distance_r13_lines
from ..snap.t3mlite import simplex # type: ignore
def _are_parallel_light_vectors(a, b, epsilon):
    for i in range(1, 4):
        if not abs(a[i] / a[0] - b[i] / b[0]) < epsilon:
            return False
    return True