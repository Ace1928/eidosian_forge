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
def _find_repeated(expr):
    if not isinstance(expr, (Basic, Unevaluated)):
        return
    if isinstance(expr, RootOf):
        return
    if isinstance(expr, Basic) and (expr.is_Atom or expr.is_Order or isinstance(expr, (MatrixSymbol, MatrixElement))):
        if expr.is_Symbol:
            excluded_symbols.add(expr)
        return
    if iterable(expr):
        args = expr
    else:
        if expr in seen_subexp:
            for ign in ignore:
                if ign in expr.free_symbols:
                    break
            else:
                to_eliminate.add(expr)
                return
        seen_subexp.add(expr)
        if expr in opt_subs:
            expr = opt_subs[expr]
        args = expr.args
    list(map(_find_repeated, args))