from __future__ import annotations
import itertools
from sympy import SYMPY_DEBUG
from sympy.core import S, Expr
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.exprtools import factor_terms
from sympy.core.function import (expand, expand_mul, expand_power_base,
from sympy.core.mul import Mul
from sympy.core.numbers import ilcm, Rational, pi
from sympy.core.relational import Eq, Ne, _canonical_coeff
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Dummy, symbols, Wild, Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (re, im, arg, Abs, sign,
from sympy.functions.elementary.exponential import exp, exp_polar, log
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.hyperbolic import (cosh, sinh,
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold
from sympy.functions.elementary.trigonometric import (cos, sin, sinc,
from sympy.functions.special.bessel import besselj, bessely, besseli, besselk
from sympy.functions.special.delta_functions import DiracDelta, Heaviside
from sympy.functions.special.elliptic_integrals import elliptic_k, elliptic_e
from sympy.functions.special.error_functions import (erf, erfc, erfi, Ei,
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper, meijerg
from sympy.functions.special.singularity_functions import SingularityFunction
from .integrals import Integral
from sympy.logic.boolalg import And, Or, BooleanAtom, Not, BooleanFunction
from sympy.polys import cancel, factor
from sympy.utilities.iterables import multiset_partitions
from sympy.utilities.misc import debug as _debug
from sympy.utilities.misc import debugf as _debugf
from sympy.utilities.timeutils import timethis
@cacheit
@timeit
def _rewrite_single(f, x, recursive=True):
    """
    Try to rewrite f as a sum of single G functions of the form
    C*x**s*G(a*x**b), where b is a rational number and C is independent of x.
    We guarantee that result.argument.as_coeff_mul(x) returns (a, (x**b,))
    or (a, ()).
    Returns a list of tuples (C, s, G) and a condition cond.
    Returns None on failure.
    """
    from .transforms import mellin_transform, inverse_mellin_transform, IntegralTransformError, MellinTransformStripError
    global _lookup_table
    if not _lookup_table:
        _lookup_table = {}
        _create_lookup_table(_lookup_table)
    if isinstance(f, meijerg):
        coeff, m = factor(f.argument, x).as_coeff_mul(x)
        if len(m) > 1:
            return None
        m = m[0]
        if m.is_Pow:
            if m.base != x or not m.exp.is_Rational:
                return None
        elif m != x:
            return None
        return ([(1, 0, meijerg(f.an, f.aother, f.bm, f.bother, coeff * m))], True)
    f_ = f
    f = f.subs(x, z)
    t = _mytype(f, z)
    if t in _lookup_table:
        l = _lookup_table[t]
        for formula, terms, cond, hint in l:
            subs = f.match(formula, old=True)
            if subs:
                subs_ = {}
                for fro, to in subs.items():
                    subs_[fro] = unpolarify(polarify(to, lift=True), exponents_only=True)
                subs = subs_
                if not isinstance(hint, bool):
                    hint = hint.subs(subs)
                if hint == False:
                    continue
                if not isinstance(cond, (bool, BooleanAtom)):
                    cond = unpolarify(cond.subs(subs))
                if _eval_cond(cond) == False:
                    continue
                if not isinstance(terms, list):
                    terms = terms(subs)
                res = []
                for fac, g in terms:
                    r1 = _get_coeff_exp(unpolarify(fac.subs(subs).subs(z, x), exponents_only=True), x)
                    try:
                        g = g.subs(subs).subs(z, x)
                    except ValueError:
                        continue
                    if Tuple(*r1 + (g,)).has(S.Infinity, S.ComplexInfinity, S.NegativeInfinity):
                        continue
                    g = meijerg(g.an, g.aother, g.bm, g.bother, unpolarify(g.argument, exponents_only=True))
                    res.append(r1 + (g,))
                if res:
                    return (res, cond)
    if not recursive:
        return None
    _debug('Trying recursive Mellin transform method.')

    def my_imt(F, s, x, strip):
        """ Calling simplify() all the time is slow and not helpful, since
            most of the time it only factors things in a way that has to be
            un-done anyway. But sometimes it can remove apparent poles. """
        try:
            return inverse_mellin_transform(F, s, x, strip, as_meijerg=True, needeval=True)
        except MellinTransformStripError:
            from sympy.simplify import simplify
            return inverse_mellin_transform(simplify(cancel(expand(F))), s, x, strip, as_meijerg=True, needeval=True)
    f = f_
    s = _dummy('s', 'rewrite-single', f)

    def my_integrator(f, x):
        r = _meijerint_definite_4(f, x, only_double=True)
        if r is not None:
            from sympy.simplify import hyperexpand
            res, cond = r
            res = _my_unpolarify(hyperexpand(res, rewrite='nonrepsmall'))
            return Piecewise((res, cond), (Integral(f, (x, S.Zero, S.Infinity)), True))
        return Integral(f, (x, S.Zero, S.Infinity))
    try:
        F, strip, _ = mellin_transform(f, x, s, integrator=my_integrator, simplify=False, needeval=True)
        g = my_imt(F, s, x, strip)
    except IntegralTransformError:
        g = None
    if g is None:
        a = _dummy_('a', 'rewrite-single')
        if a not in f.free_symbols and _is_analytic(f, x):
            try:
                F, strip, _ = mellin_transform(f.subs(x, a * x), x, s, integrator=my_integrator, needeval=True, simplify=False)
                g = my_imt(F, s, x, strip).subs(a, 1)
            except IntegralTransformError:
                g = None
    if g is None or g.has(S.Infinity, S.NaN, S.ComplexInfinity):
        _debug('Recursive Mellin transform failed.')
        return None
    args = Add.make_args(g)
    res = []
    for f in args:
        c, m = f.as_coeff_mul(x)
        if len(m) > 1:
            raise NotImplementedError('Unexpected form...')
        g = m[0]
        a, b = _get_coeff_exp(g.argument, x)
        res += [(c, 0, meijerg(g.an, g.aother, g.bm, g.bother, unpolarify(polarify(a, lift=True), exponents_only=True) * x ** b))]
    _debug('Recursive Mellin transform worked:', g)
    return (res, True)