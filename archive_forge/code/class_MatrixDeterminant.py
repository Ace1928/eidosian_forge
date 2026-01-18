import mpmath as mp
from collections.abc import Callable
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.function import diff
from sympy.core.expr import Expr
from sympy.core.kind import _NumberKind, UndefinedKind
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol, uniquely_named_symbol
from sympy.core.sympify import sympify, _sympify
from sympy.functions.combinatorial.factorials import binomial, factorial
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import Max, Min, sqrt
from sympy.functions.special.tensor_functions import KroneckerDelta, LeviCivita
from sympy.polys import cancel
from sympy.printing import sstr
from sympy.printing.defaults import Printable
from sympy.printing.str import StrPrinter
from sympy.utilities.iterables import flatten, NotIterable, is_sequence, reshape
from sympy.utilities.misc import as_int, filldedent
from .common import (
from .utilities import _iszero, _is_zero_after_expand_mul, _simplify
from .determinant import (
from .reductions import _is_echelon, _echelon_form, _rank, _rref
from .subspaces import _columnspace, _nullspace, _rowspace, _orthogonalize
from .eigen import (
from .decompositions import (
from .graph import (
from .solvers import (
from .inverse import (
class MatrixDeterminant(MatrixCommon):
    """Provides basic matrix determinant operations. Should not be instantiated
    directly. See ``determinant.py`` for their implementations."""

    def _eval_det_bareiss(self, iszerofunc=_is_zero_after_expand_mul):
        return _det_bareiss(self, iszerofunc=iszerofunc)

    def _eval_det_berkowitz(self):
        return _det_berkowitz(self)

    def _eval_det_lu(self, iszerofunc=_iszero, simpfunc=None):
        return _det_LU(self, iszerofunc=iszerofunc, simpfunc=simpfunc)

    def _eval_determinant(self):
        return _det(self)

    def adjugate(self, method='berkowitz'):
        return _adjugate(self, method=method)

    def charpoly(self, x='lambda', simplify=_simplify):
        return _charpoly(self, x=x, simplify=simplify)

    def cofactor(self, i, j, method='berkowitz'):
        return _cofactor(self, i, j, method=method)

    def cofactor_matrix(self, method='berkowitz'):
        return _cofactor_matrix(self, method=method)

    def det(self, method='bareiss', iszerofunc=None):
        return _det(self, method=method, iszerofunc=iszerofunc)

    def per(self):
        return _per(self)

    def minor(self, i, j, method='berkowitz'):
        return _minor(self, i, j, method=method)

    def minor_submatrix(self, i, j):
        return _minor_submatrix(self, i, j)
    _find_reasonable_pivot.__doc__ = _find_reasonable_pivot.__doc__
    _find_reasonable_pivot_naive.__doc__ = _find_reasonable_pivot_naive.__doc__
    _eval_det_bareiss.__doc__ = _det_bareiss.__doc__
    _eval_det_berkowitz.__doc__ = _det_berkowitz.__doc__
    _eval_det_lu.__doc__ = _det_LU.__doc__
    _eval_determinant.__doc__ = _det.__doc__
    adjugate.__doc__ = _adjugate.__doc__
    charpoly.__doc__ = _charpoly.__doc__
    cofactor.__doc__ = _cofactor.__doc__
    cofactor_matrix.__doc__ = _cofactor_matrix.__doc__
    det.__doc__ = _det.__doc__
    per.__doc__ = _per.__doc__
    minor.__doc__ = _minor.__doc__
    minor_submatrix.__doc__ = _minor_submatrix.__doc__