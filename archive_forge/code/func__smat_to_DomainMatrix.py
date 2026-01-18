from collections import defaultdict
from operator import index as index_
from sympy.core.expr import Expr
from sympy.core.kind import Kind, NumberKind, UndefinedKind
from sympy.core.numbers import Integer, Rational
from sympy.core.sympify import _sympify, SympifyError
from sympy.core.singleton import S
from sympy.polys.domains import ZZ, QQ, EXRAW
from sympy.polys.matrices import DomainMatrix
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import filldedent
from .common import classof
from .matrices import MatrixBase, MatrixKind, ShapeError
@classmethod
def _smat_to_DomainMatrix(cls, rows, cols, smat):
    elements_dod = defaultdict(dict)
    for (i, j), element in smat.items():
        if element != 0:
            elements_dod[i][j] = element
    types = set(map(type, smat.values()))
    rep = cls._dod_to_DomainMatrix(rows, cols, elements_dod, types)
    return rep