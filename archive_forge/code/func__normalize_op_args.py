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
def _normalize_op_args(self, op, col, k, col1, col2, error_str='col'):
    """Validate the arguments for a row/column operation.  ``error_str``
        can be one of "row" or "col" depending on the arguments being parsed."""
    if op not in ['n->kn', 'n<->m', 'n->n+km']:
        raise ValueError("Unknown {} operation '{}'. Valid col operations are 'n->kn', 'n<->m', 'n->n+km'".format(error_str, op))
    self_cols = self.cols if error_str == 'col' else self.rows
    if op == 'n->kn':
        col = col if col is not None else col1
        if col is None or k is None:
            raise ValueError("For a {0} operation 'n->kn' you must provide the kwargs `{0}` and `k`".format(error_str))
        if not 0 <= col < self_cols:
            raise ValueError("This matrix does not have a {} '{}'".format(error_str, col))
    elif op == 'n<->m':
        cols = {col, k, col1, col2}.difference([None])
        if len(cols) > 2:
            cols = {col, col1, col2}.difference([None])
        if len(cols) != 2:
            raise ValueError("For a {0} operation 'n<->m' you must provide the kwargs `{0}1` and `{0}2`".format(error_str))
        col1, col2 = cols
        if not 0 <= col1 < self_cols:
            raise ValueError("This matrix does not have a {} '{}'".format(error_str, col1))
        if not 0 <= col2 < self_cols:
            raise ValueError("This matrix does not have a {} '{}'".format(error_str, col2))
    elif op == 'n->n+km':
        col = col1 if col is None else col
        col2 = col1 if col2 is None else col2
        if col is None or col2 is None or k is None:
            raise ValueError("For a {0} operation 'n->n+km' you must provide the kwargs `{0}`, `k`, and `{0}2`".format(error_str))
        if col == col2:
            raise ValueError("For a {0} operation 'n->n+km' `{0}` and `{0}2` must be different.".format(error_str))
        if not 0 <= col < self_cols:
            raise ValueError("This matrix does not have a {} '{}'".format(error_str, col))
        if not 0 <= col2 < self_cols:
            raise ValueError("This matrix does not have a {} '{}'".format(error_str, col2))
    else:
        raise ValueError('invalid operation %s' % repr(op))
    return (op, col, k, col1, col2)