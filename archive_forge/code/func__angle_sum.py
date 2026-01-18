from .verificationError import *
from snappy.snap import t3mlite as t3m
from sage.all import vector, matrix, prod, exp, RealDoubleField, sqrt
import sage.all
def _angle_sum(self, edge):
    return sum([self._angle_at_corner(corner) for corner in edge.Corners])