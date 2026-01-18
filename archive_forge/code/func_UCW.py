from sympy.combinatorics.permutations import Permutation
from sympy.core.symbol import symbols
from sympy.matrices import Matrix
from sympy.utilities.iterables import variations, rotate_left
def UCW(r=1):
    for _ in range(r):
        cw(U)
        ccw(D)
        t = faces[F]
        faces[F] = faces[R]
        faces[R] = faces[B]
        faces[B] = faces[L]
        faces[L] = t