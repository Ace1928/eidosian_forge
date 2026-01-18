from collections import defaultdict
from itertools import product
from functools import reduce
from math import prod
from sympy import SYMPY_DEBUG
from sympy.core import (S, Dummy, symbols, sympify, Tuple, expand, I, pi, Mul,
from sympy.core.mod import Mod
from sympy.core.sorting import default_sort_key
from sympy.functions import (exp, sqrt, root, log, lowergamma, cos,
from sympy.functions.elementary.complexes import polarify, unpolarify
from sympy.functions.special.hyper import (hyper, HyperRep_atanh,
from sympy.matrices import Matrix, eye, zeros
from sympy.polys import apart, poly, Poly
from sympy.series import residue
from sympy.simplify.powsimp import powdenest
from sympy.utilities.iterables import sift
def _meijergexpand(func, z0, allow_hyper=False, rewrite='default', place=None):
    """
    Try to find an expression for the Meijer G function specified
    by the G_Function ``func``. If ``allow_hyper`` is True, then returning
    an expression in terms of hypergeometric functions is allowed.

    Currently this just does Slater's theorem.
    If expansions exist both at zero and at infinity, ``place``
    can be set to ``0`` or ``zoo`` for the preferred choice.
    """
    global _meijercollection
    if _meijercollection is None:
        _meijercollection = MeijerFormulaCollection()
    if rewrite == 'default':
        rewrite = None
    func0 = func
    debug('Try to expand Meijer G function corresponding to ', func)
    z = Dummy('z')
    func, ops = reduce_order_meijer(func)
    if ops:
        debug('  Reduced order to ', func)
    else:
        debug('  Could not reduce order.')
    f = _meijercollection.lookup_origin(func)
    if f is not None:
        debug('  Found a Meijer G formula: ', f.func)
        ops += devise_plan_meijer(f.func, func, z)
        C = apply_operators(f.C.subs(f.z, z), ops, make_derivative_operator(f.M.subs(f.z, z), z))
        C = C.applyfunc(make_simp(z))
        r = C * f.B.subs(f.z, z)
        r = r[0].subs(z, z0)
        return powdenest(r, polar=True)
    debug("  Could not find a direct formula. Trying Slater's theorem.")

    def can_do(pbm, pap):
        """ Test if slater applies. """
        for i in pbm:
            if len(pbm[i]) > 1:
                l = 0
                if i in pap:
                    l = len(pap[i])
                if l + 1 < len(pbm[i]):
                    return False
        return True

    def do_slater(an, bm, ap, bq, z, zfinal):
        func = G_Function(an, bm, ap, bq)
        _, pbm, pap, _ = func.compute_buckets()
        if not can_do(pbm, pap):
            return (S.Zero, False)
        cond = len(an) + len(ap) < len(bm) + len(bq)
        if len(an) + len(ap) == len(bm) + len(bq):
            cond = abs(z) < 1
        if cond is False:
            return (S.Zero, False)
        res = S.Zero
        for m in pbm:
            if len(pbm[m]) == 1:
                bh = pbm[m][0]
                fac = 1
                bo = list(bm)
                bo.remove(bh)
                for bj in bo:
                    fac *= gamma(bj - bh)
                for aj in an:
                    fac *= gamma(1 + bh - aj)
                for bj in bq:
                    fac /= gamma(1 + bh - bj)
                for aj in ap:
                    fac /= gamma(aj - bh)
                nap = [1 + bh - a for a in list(an) + list(ap)]
                nbq = [1 + bh - b for b in list(bo) + list(bq)]
                k = polar_lift(S.NegativeOne ** (len(ap) - len(bm)))
                harg = k * zfinal
                premult = (t / k) ** bh
                hyp = _hyperexpand(Hyper_Function(nap, nbq), harg, ops, t, premult, bh, rewrite=None)
                res += fac * hyp
            else:
                b_ = pbm[m][0]
                ki = [bi - b_ for bi in pbm[m][1:]]
                u = len(ki)
                li = [ai - b_ for ai in pap[m][:u + 1]]
                bo = list(bm)
                for b in pbm[m]:
                    bo.remove(b)
                ao = list(ap)
                for a in pap[m][:u]:
                    ao.remove(a)
                lu = li[-1]
                di = [l - k for l, k in zip(li, ki)]
                s = Dummy('s')
                integrand = z ** s
                for b in bm:
                    if not Mod(b, 1) and b.is_Number:
                        b = int(round(b))
                    integrand *= gamma(b - s)
                for a in an:
                    integrand *= gamma(1 - a + s)
                for b in bq:
                    integrand /= gamma(1 - b + s)
                for a in ap:
                    integrand /= gamma(a - s)
                integrand = expand_func(integrand)
                for r in range(int(round(lu))):
                    resid = residue(integrand, s, b_ + r)
                    resid = apply_operators(resid, ops, lambda f: z * f.diff(z))
                    res -= resid
                au = b_ + lu
                k = polar_lift(S.NegativeOne ** (len(ao) + len(bo) + 1))
                harg = k * zfinal
                premult = (t / k) ** au
                nap = [1 + au - a for a in list(an) + list(ap)] + [1]
                nbq = [1 + au - b for b in list(bm) + list(bq)]
                hyp = _hyperexpand(Hyper_Function(nap, nbq), harg, ops, t, premult, au, rewrite=None)
                C = S.NegativeOne ** lu / factorial(lu)
                for i in range(u):
                    C *= S.NegativeOne ** di[i] / rf(lu - li[i] + 1, di[i])
                for a in an:
                    C *= gamma(1 - a + au)
                for b in bo:
                    C *= gamma(b - au)
                for a in ao:
                    C /= gamma(a - au)
                for b in bq:
                    C /= gamma(1 - b + au)
                res += C * hyp
        return (res, cond)
    t = Dummy('t')
    slater1, cond1 = do_slater(func.an, func.bm, func.ap, func.bq, z, z0)

    def tr(l):
        return [1 - x for x in l]
    for op in ops:
        op._poly = Poly(op._poly.subs({z: 1 / t, _x: -_x}), _x)
    slater2, cond2 = do_slater(tr(func.bm), tr(func.an), tr(func.bq), tr(func.ap), t, 1 / z0)
    slater1 = powdenest(slater1.subs(z, z0), polar=True)
    slater2 = powdenest(slater2.subs(t, 1 / z0), polar=True)
    if not isinstance(cond2, bool):
        cond2 = cond2.subs(t, 1 / z)
    m = func(z)
    if m.delta > 0 or (m.delta == 0 and len(m.ap) == len(m.bq) and ((re(m.nu) < -1) is not False) and (polar_lift(z0) == polar_lift(1))):
        if cond1 is not False:
            cond1 = True
        if cond2 is not False:
            cond2 = True
    if cond1 is True:
        slater1 = slater1.rewrite(rewrite or 'nonrep')
    else:
        slater1 = slater1.rewrite(rewrite or 'nonrepsmall')
    if cond2 is True:
        slater2 = slater2.rewrite(rewrite or 'nonrep')
    else:
        slater2 = slater2.rewrite(rewrite or 'nonrepsmall')
    if cond1 is not False and cond2 is not False:
        if place == 0:
            cond2 = False
        if place == zoo:
            cond1 = False
    if not isinstance(cond1, bool):
        cond1 = cond1.subs(z, z0)
    if not isinstance(cond2, bool):
        cond2 = cond2.subs(z, z0)

    def weight(expr, cond):
        if cond is True:
            c0 = 0
        elif cond is False:
            c0 = 1
        else:
            c0 = 2
        if expr.has(oo, zoo, -oo, nan):
            c0 = 3
        return (c0, expr.count(hyper), expr.count_ops())
    w1 = weight(slater1, cond1)
    w2 = weight(slater2, cond2)
    if min(w1, w2) <= (0, 1, oo):
        if w1 < w2:
            return slater1
        else:
            return slater2
    if max(w1[0], w2[0]) <= 1 and max(w1[1], w2[1]) <= 1:
        return Piecewise((slater1, cond1), (slater2, cond2), (func0(z0), True))
    r = Piecewise((slater1, cond1), (slater2, cond2), (func0(z0), True))
    if r.has(hyper) and (not allow_hyper):
        debug('  Could express using hypergeometric functions, but not allowed.')
    if not r.has(hyper) or allow_hyper:
        return r
    return func0(z0)