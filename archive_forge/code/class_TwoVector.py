from builtins import range
import sys
from math import sqrt, cos, sin, atan2, pi
class TwoVector(tuple):

    def __new__(cls, x, y):
        return tuple.__new__(cls, (x, y))

    def __add__(self, other):
        return TwoVector(self[0] + other[0], self[1] + other[1])

    def __sub__(self, other):
        return TwoVector(self[0] - other[0], self[1] - other[1])

    def __rmul__(self, scalar):
        return TwoVector(scalar * self[0], scalar * self[1])

    def __xor__(self, other):
        return self[0] * other[1] - self[1] * other[0]

    def __abs__(self):
        return sqrt(self[0] * self[0] + self[1] * self[1])

    def angle(self):
        return atan2(self[1], self[0])

    def unit(self):
        return 1 / abs(self) * self