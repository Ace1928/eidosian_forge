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
def _dod_to_DomainMatrix(cls, rows, cols, dod, types):
    if not all((issubclass(typ, Expr) for typ in types)):
        sympy_deprecation_warning('\n                non-Expr objects in a Matrix is deprecated. Matrix represents\n                a mathematical matrix. To represent a container of non-numeric\n                entities, Use a list of lists, TableForm, NumPy array, or some\n                other data structure instead.\n                ', deprecated_since_version='1.9', active_deprecations_target='deprecated-non-expr-in-matrix', stacklevel=6)
    rep = DomainMatrix(dod, (rows, cols), EXRAW)
    if all((issubclass(typ, Rational) for typ in types)):
        if all((issubclass(typ, Integer) for typ in types)):
            rep = rep.convert_to(ZZ)
        else:
            rep = rep.convert_to(QQ)
    return rep