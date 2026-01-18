import sys
from math import sin, cos, acos, sqrt
import numpy as np
from qiskit.exceptions import MissingOptionalLibraryError
def _axisangle_to_q(self, theta, v):
    """Convert axis and angle to quaternion"""
    x = v[0]
    y = v[1]
    z = v[2]
    w = cos(theta / 2.0)
    x = x * sin(theta / 2.0)
    y = y * sin(theta / 2.0)
    z = z * sin(theta / 2.0)
    self._val = np.array([w, x, y, z])