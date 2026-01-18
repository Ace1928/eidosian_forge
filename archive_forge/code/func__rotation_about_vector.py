import numpy as np
from matplotlib import _api
def _rotation_about_vector(v, angle):
    """
    Produce a rotation matrix for an angle in radians about a vector.
    """
    vx, vy, vz = v / np.linalg.norm(v)
    s = np.sin(angle)
    c = np.cos(angle)
    t = 2 * np.sin(angle / 2) ** 2
    R = np.array([[t * vx * vx + c, t * vx * vy - vz * s, t * vx * vz + vy * s], [t * vy * vx + vz * s, t * vy * vy + c, t * vy * vz - vx * s], [t * vz * vx - vy * s, t * vz * vy + vx * s, t * vz * vz + c]])
    return R