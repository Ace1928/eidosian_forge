from snappy.SnapPy import matrix, vector
from snappy.hyperboloid import (r13_dot,
def R13_plane_from_R13_light_vectors(pts):
    return R13_normalise(unnormalised_plane_eqn_from_r13_points(pts))