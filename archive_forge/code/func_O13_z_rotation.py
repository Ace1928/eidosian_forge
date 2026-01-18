from snappy.SnapPy import matrix, vector
from snappy.hyperboloid import (r13_dot,
def O13_z_rotation(angle):
    """
    SO(1,3)-matrix corresponding to a rotation about the z-Axis
    by angle (in radians).
    """
    c = angle.cos()
    s = angle.sin()
    return matrix([[1, 0, 0, 0], [0, c, s, 0], [0, -s, c, 0], [0, 0, 0, 1]], ring=angle.parent())