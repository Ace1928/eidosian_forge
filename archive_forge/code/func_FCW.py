from sympy.combinatorics.permutations import Permutation
from sympy.core.symbol import symbols
from sympy.matrices import Matrix
from sympy.utilities.iterables import variations, rotate_left
def FCW(r=1):
    for _ in range(r):
        cw(F)
        ccw(B)
        cw(U)
        t = faces[U]
        cw(L)
        faces[U] = faces[L]
        cw(D)
        faces[L] = faces[D]
        cw(R)
        faces[D] = faces[R]
        faces[R] = t