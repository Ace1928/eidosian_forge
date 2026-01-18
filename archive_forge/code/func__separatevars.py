from collections import defaultdict
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core import (Basic, S, Add, Mul, Pow, Symbol, sympify,
from sympy.core.exprtools import factor_nc
from sympy.core.parameters import global_parameters
from sympy.core.function import (expand_log, count_ops, _mexpand,
from sympy.core.numbers import Float, I, pi, Rational
from sympy.core.relational import Relational
from sympy.core.rules import Transform
from sympy.core.sorting import ordered
from sympy.core.sympify import _sympify
from sympy.core.traversal import bottom_up as _bottom_up, walk as _walk
from sympy.functions import gamma, exp, sqrt, log, exp_polar, re
from sympy.functions.combinatorial.factorials import CombinatorialFunction
from sympy.functions.elementary.complexes import unpolarify, Abs, sign
from sympy.functions.elementary.exponential import ExpBase
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.piecewise import (Piecewise, piecewise_fold,
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.functions.special.bessel import (BesselBase, besselj, besseli,
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.integrals.integrals import Integral
from sympy.matrices.expressions import (MatrixExpr, MatAdd, MatMul,
from sympy.polys import together, cancel, factor
from sympy.polys.numberfields.minpoly import _is_sum_surds, _minimal_polynomial_sq
from sympy.simplify.combsimp import combsimp
from sympy.simplify.cse_opts import sub_pre, sub_post
from sympy.simplify.hyperexpand import hyperexpand
from sympy.simplify.powsimp import powsimp
from sympy.simplify.radsimp import radsimp, fraction, collect_abs
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.simplify.trigsimp import trigsimp, exptrigsimp
from sympy.utilities.decorator import deprecated
from sympy.utilities.iterables import has_variety, sift, subsets, iterable
from sympy.utilities.misc import as_int
import mpmath
def _separatevars(expr, force):
    if isinstance(expr, Abs):
        arg = expr.args[0]
        if arg.is_Mul and (not arg.is_number):
            s = separatevars(arg, dict=True, force=force)
            if s is not None:
                return Mul(*map(expr.func, s.values()))
            else:
                return expr
    if len(expr.free_symbols) < 2:
        return expr
    if expr.is_Mul:
        args = list(expr.args)
        changed = False
        for i, a in enumerate(args):
            args[i] = separatevars(a, force)
            changed = changed or args[i] != a
        if changed:
            expr = expr.func(*args)
        return expr
    if expr.is_Pow and expr.base != S.Exp1:
        expr = Pow(separatevars(expr.base, force=force), expr.exp)
    expr = expr.expand(mul=False, multinomial=False, force=force)
    _expr, reps = posify(expr) if force else (expr, {})
    expr = factor(_expr).subs(reps)
    if not expr.is_Add:
        return expr
    args = list(expr.args)
    commonc = args[0].args_cnc(cset=True, warn=False)[0]
    for i in args[1:]:
        commonc &= i.args_cnc(cset=True, warn=False)[0]
    commonc = Mul(*commonc)
    commonc = commonc.as_coeff_Mul()[1]
    commonc_set = commonc.args_cnc(cset=True, warn=False)[0]
    for i, a in enumerate(args):
        c, nc = a.args_cnc(cset=True, warn=False)
        c = c - commonc_set
        args[i] = Mul(*c) * Mul(*nc)
    nonsepar = Add(*args)
    if len(nonsepar.free_symbols) > 1:
        _expr = nonsepar
        _expr, reps = posify(_expr) if force else (_expr, {})
        _expr = factor(_expr).subs(reps)
        if not _expr.is_Add:
            nonsepar = _expr
    return commonc * nonsepar