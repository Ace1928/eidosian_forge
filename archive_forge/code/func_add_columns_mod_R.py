from collections import defaultdict
from .domainmatrix import DomainMatrix
from .exceptions import DMDomainError, DMShapeError
from sympy.ntheory.modular import symmetric_residue
from sympy.polys.domains import QQ, ZZ
def add_columns_mod_R(m, R, i, j, a, b, c, d):
    for k in range(len(m)):
        e = m[k][i]
        m[k][i] = symmetric_residue((a * e + b * m[k][j]) % R, R)
        m[k][j] = symmetric_residue((c * e + d * m[k][j]) % R, R)