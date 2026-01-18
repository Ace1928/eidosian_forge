from sympy.core.basic import Basic
from sympy.core import (S, Expr, Integer, Float, I, oo, Add, Lambda,
from sympy.core.cache import cacheit
from sympy.core.relational import is_le
from sympy.core.sorting import ordered
from sympy.polys.domains import QQ
from sympy.polys.polyerrors import (
from sympy.polys.polyfuncs import symmetrize, viete
from sympy.polys.polyroots import (
from sympy.polys.polytools import Poly, PurePoly, factor
from sympy.polys.rationaltools import together
from sympy.polys.rootisolation import (
from sympy.utilities import lambdify, public, sift, numbered_symbols
from mpmath import mpf, mpc, findroot, workprec
from mpmath.libmp.libmpf import dps_to_prec, prec_to_dps
from sympy.multipledispatch import dispatch
from itertools import chain
@classmethod
def _rational_case(cls, poly, func):
    """Handle the rational function case. """
    roots = symbols('r:%d' % poly.degree())
    var, expr = (func.variables[0], func.expr)
    f = sum((expr.subs(var, r) for r in roots))
    p, q = together(f).as_numer_denom()
    domain = QQ[roots]
    p = p.expand()
    q = q.expand()
    try:
        p = Poly(p, domain=domain, expand=False)
    except GeneratorsNeeded:
        p, p_coeff = (None, (p,))
    else:
        p_monom, p_coeff = zip(*p.terms())
    try:
        q = Poly(q, domain=domain, expand=False)
    except GeneratorsNeeded:
        q, q_coeff = (None, (q,))
    else:
        q_monom, q_coeff = zip(*q.terms())
    coeffs, mapping = symmetrize(p_coeff + q_coeff, formal=True)
    formulas, values = (viete(poly, roots), [])
    for (sym, _), (_, val) in zip(mapping, formulas):
        values.append((sym, val))
    for i, (coeff, _) in enumerate(coeffs):
        coeffs[i] = coeff.subs(values)
    n = len(p_coeff)
    p_coeff = coeffs[:n]
    q_coeff = coeffs[n:]
    if p is not None:
        p = Poly(dict(zip(p_monom, p_coeff)), *p.gens).as_expr()
    else:
        p, = p_coeff
    if q is not None:
        q = Poly(dict(zip(q_monom, q_coeff)), *q.gens).as_expr()
    else:
        q, = q_coeff
    return factor(p / q)