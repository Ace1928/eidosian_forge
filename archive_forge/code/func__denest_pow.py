from collections import defaultdict
from functools import reduce
from math import prod
from sympy.core.function import expand_log, count_ops, _coeff_isneg
from sympy.core import sympify, Basic, Dummy, S, Add, Mul, Pow, expand_mul, factor_terms
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.numbers import Integer, Rational
from sympy.core.mul import _keep_coeff
from sympy.core.rules import Transform
from sympy.functions import exp_polar, exp, log, root, polarify, unpolarify
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.polys import lcm, gcd
from sympy.ntheory.factor_ import multiplicity
def _denest_pow(eq):
    """
    Denest powers.

    This is a helper function for powdenest that performs the actual
    transformation.
    """
    from sympy.simplify.simplify import logcombine
    b, e = eq.as_base_exp()
    if b.is_Pow or (isinstance(b, exp) and e != 1):
        new = b._eval_power(e)
        if new is not None:
            eq = new
            b, e = new.as_base_exp()
    if b is S.Exp1 and e.is_Mul:
        logs = []
        other = []
        for ei in e.args:
            if any((isinstance(ai, log) for ai in Add.make_args(ei))):
                logs.append(ei)
            else:
                other.append(ei)
        logs = logcombine(Mul(*logs))
        return Pow(exp(logs), Mul(*other))
    _, be = b.as_base_exp()
    if be is S.One and (not (b.is_Mul or (b.is_Rational and b.q != 1) or b.is_positive)):
        return eq
    polars, nonpolars = ([], [])
    for bb in Mul.make_args(b):
        if bb.is_polar:
            polars.append(bb.as_base_exp())
        else:
            nonpolars.append(bb)
    if len(polars) == 1 and (not polars[0][0].is_Mul):
        return Pow(polars[0][0], polars[0][1] * e) * powdenest(Mul(*nonpolars) ** e)
    elif polars:
        return Mul(*[powdenest(bb ** (ee * e)) for bb, ee in polars]) * powdenest(Mul(*nonpolars) ** e)
    if b.is_Integer:
        logb = expand_log(log(b))
        if logb.is_Mul:
            c, logb = logb.args
            e *= c
            base = logb.args[0]
            return Pow(base, e)
    if not b.is_Mul or any((s.is_Atom for s in Mul.make_args(b))):
        return eq

    def nc_gcd(aa, bb):
        a, b = [i.as_coeff_Mul() for i in [aa, bb]]
        c = gcd(a[0], b[0]).as_numer_denom()[0]
        g = Mul(*a[1].args_cnc(cset=True)[0] & b[1].args_cnc(cset=True)[0])
        return _keep_coeff(c, g)
    glogb = expand_log(log(b))
    if glogb.is_Add:
        args = glogb.args
        g = reduce(nc_gcd, args)
        if g != 1:
            cg, rg = g.as_coeff_Mul()
            glogb = _keep_coeff(cg, rg * Add(*[a / g for a in args]))
    if isinstance(glogb, log) or not glogb.is_Mul:
        if glogb.args[0].is_Pow or isinstance(glogb.args[0], exp):
            glogb = _denest_pow(glogb.args[0])
            if (abs(glogb.exp) < 1) == True:
                return Pow(glogb.base, glogb.exp * e)
        return eq
    add = []
    other = []
    for a in glogb.args:
        if a.is_Add:
            add.append(a)
        else:
            other.append(a)
    return Pow(exp(logcombine(Mul(*add))), e * Mul(*other))