from .verificationError import *
from snappy.snap import t3mlite as t3m
from sage.all import vector, matrix, prod, exp, RealDoubleField, sqrt
import sage.all
def _find_max(m, rows_left, cols_left):
    v = -1
    for r in rows_left:
        for c in cols_left:
            a = abs(m[r, c])
            if a > v:
                v, row, col = (a, r, c)
    return (row, col)