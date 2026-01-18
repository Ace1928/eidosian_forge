from .verificationError import *
from snappy.snap import t3mlite as t3m
from sage.all import vector, matrix, prod, exp, RealDoubleField, sqrt
import sage.all
def _angle_at_corner(self, corner):
    i, j = [k for k, z in enumerate(t3m.ZeroSubsimplices) if not z & corner.Subsimplex]
    return self.dihedral_angles[corner.Tetrahedron.Index][i][j]