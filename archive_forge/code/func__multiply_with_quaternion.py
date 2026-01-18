import sys
from math import sin, cos, acos, sqrt
import numpy as np
from qiskit.exceptions import MissingOptionalLibraryError
def _multiply_with_quaternion(self, q_2):
    """Multiplication of quaternion with quaternion"""
    w_1, x_1, y_1, z_1 = self._val
    w_2, x_2, y_2, z_2 = q_2._val
    w = w_1 * w_2 - x_1 * x_2 - y_1 * y_2 - z_1 * z_2
    x = w_1 * x_2 + x_1 * w_2 + y_1 * z_2 - z_1 * y_2
    y = w_1 * y_2 + y_1 * w_2 + z_1 * x_2 - x_1 * z_2
    z = w_1 * z_2 + z_1 * w_2 + x_1 * y_2 - y_1 * x_2
    result = _Quaternion.from_value(np.array((w, x, y, z)))
    return result