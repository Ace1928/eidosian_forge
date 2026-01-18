from .verificationError import *
from snappy.snap import t3mlite as t3m
from sage.all import vector, matrix, prod, exp, RealDoubleField, sqrt
import sage.all
def _cofactor_derivative(cofactor_matrices, i, j, m, n):
    return _one_cofactor_derivative(cofactor_matrices, i, j, m, n) + _one_cofactor_derivative(cofactor_matrices, i, j, n, m)