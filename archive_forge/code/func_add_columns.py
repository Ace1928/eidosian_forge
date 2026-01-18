from collections import defaultdict
from .domainmatrix import DomainMatrix
from .exceptions import DMDomainError, DMShapeError
from sympy.ntheory.modular import symmetric_residue
from sympy.polys.domains import QQ, ZZ
def add_columns(m, i, j, a, b, c, d):
    for k in range(len(m)):
        e = m[k][i]
        m[k][i] = a * e + b * m[k][j]
        m[k][j] = c * e + d * m[k][j]