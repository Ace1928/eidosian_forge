from collections import defaultdict
from sympy.core import Basic, Mul, Add, Pow, sympify
from sympy.core.containers import Tuple, OrderedSet
from sympy.core.exprtools import factor_terms
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import symbols, Symbol
from sympy.matrices import (MatrixBase, Matrix, ImmutableMatrix,
from sympy.matrices.expressions import (MatrixExpr, MatrixSymbol, MatMul,
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.polys.rootoftools import RootOf
from sympy.utilities.iterables import numbered_symbols, sift, \
from . import cse_opts
def _find_opts(expr):
    if not isinstance(expr, (Basic, Unevaluated)):
        return
    if expr.is_Atom or expr.is_Order:
        return
    if iterable(expr):
        list(map(_find_opts, expr))
        return
    if expr in seen_subexp:
        return expr
    seen_subexp.add(expr)
    list(map(_find_opts, expr.args))
    if not isinstance(expr, MatrixExpr) and expr.could_extract_minus_sign():
        if isinstance(expr, Add):
            neg_expr = Add(*(-i for i in expr.args))
        else:
            neg_expr = -expr
        if not neg_expr.is_Atom:
            opt_subs[expr] = Unevaluated(Mul, (S.NegativeOne, neg_expr))
            seen_subexp.add(neg_expr)
            expr = neg_expr
    if isinstance(expr, (Mul, MatMul)):
        if len(expr.args) == 1:
            collapsible_subexp.add(expr)
        else:
            muls.add(expr)
    elif isinstance(expr, (Add, MatAdd)):
        if len(expr.args) == 1:
            collapsible_subexp.add(expr)
        else:
            adds.add(expr)
    elif isinstance(expr, Inverse):
        pass
    elif isinstance(expr, (Pow, MatPow)):
        base, exp = (expr.base, expr.exp)
        if exp.could_extract_minus_sign():
            opt_subs[expr] = Unevaluated(Pow, (Pow(base, -exp), -1))