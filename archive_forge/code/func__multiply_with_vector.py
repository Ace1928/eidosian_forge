import sys
from math import sin, cos, acos, sqrt
import numpy as np
from qiskit.exceptions import MissingOptionalLibraryError
def _multiply_with_vector(self, v):
    """Multiplication of quaternion with vector"""
    q_2 = _Quaternion.from_value(np.append(0.0, v))
    return (self * q_2 * self.get_conjugate())._val[1:]