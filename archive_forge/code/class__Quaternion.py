import sys
from math import sin, cos, acos, sqrt
import numpy as np
from qiskit.exceptions import MissingOptionalLibraryError
class _Quaternion:
    """For calculating vectors on unit sphere"""

    def __init__(self):
        self._val = None

    @staticmethod
    def from_axisangle(theta, v):
        """Create quaternion from axis"""
        v = _normalize(v)
        new_quaternion = _Quaternion()
        new_quaternion._axisangle_to_q(theta, v)
        return new_quaternion

    @staticmethod
    def from_value(value):
        """Create quaternion from vector"""
        new_quaternion = _Quaternion()
        new_quaternion._val = value
        return new_quaternion

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

    def __mul__(self, b):
        """Multiplication of quaternion with quaternion or vector"""
        if isinstance(b, _Quaternion):
            return self._multiply_with_quaternion(b)
        elif isinstance(b, (list, tuple, np.ndarray)):
            if len(b) != 3:
                raise Exception(f'Input vector has invalid length {len(b)}')
            return self._multiply_with_vector(b)
        else:
            raise Exception(f'Multiplication with unknown type {type(b)}')

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

    def _multiply_with_vector(self, v):
        """Multiplication of quaternion with vector"""
        q_2 = _Quaternion.from_value(np.append(0.0, v))
        return (self * q_2 * self.get_conjugate())._val[1:]

    def get_conjugate(self):
        """Conjugation of quaternion"""
        w, x, y, z = self._val
        result = _Quaternion.from_value(np.array((w, -x, -y, -z)))
        return result

    def __repr__(self):
        theta, v = self.get_axisangle()
        return f'(({theta}; {v[0]}, {v[1]}, {v[2]}))'

    def get_axisangle(self):
        """Returns angle and vector of quaternion"""
        w, v = (self._val[0], self._val[1:])
        theta = acos(w) * 2.0
        return (theta, _normalize(v))

    def tolist(self):
        """Converts quaternion to a list"""
        return self._val.tolist()

    def vector_norm(self):
        """Calculates norm of quaternion"""
        _, v = self.get_axisangle()
        return np.linalg.norm(v)