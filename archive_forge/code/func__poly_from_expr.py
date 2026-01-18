from functools import wraps, reduce
from operator import mul
from typing import Optional
from sympy.core import (
from sympy.core.basic import Basic
from sympy.core.decorators import _sympifyit
from sympy.core.exprtools import Factors, factor_nc, factor_terms
from sympy.core.evalf import (
from sympy.core.function import Derivative
from sympy.core.mul import Mul, _keep_coeff
from sympy.core.numbers import ilcm, I, Integer, equal_valued
from sympy.core.relational import Relational, Equality
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify, _sympify
from sympy.core.traversal import preorder_traversal, bottom_up
from sympy.logic.boolalg import BooleanAtom
from sympy.polys import polyoptions as options
from sympy.polys.constructor import construct_domain
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.fglmtools import matrix_fglm
from sympy.polys.groebnertools import groebner as _groebner
from sympy.polys.monomials import Monomial
from sympy.polys.orderings import monomial_key
from sympy.polys.polyclasses import DMP, DMF, ANP
from sympy.polys.polyerrors import (
from sympy.polys.polyutils import (
from sympy.polys.rationaltools import together
from sympy.polys.rootisolation import dup_isolate_real_roots_list
from sympy.utilities import group, public, filldedent
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import iterable, sift
import sympy.polys
import mpmath
from mpmath.libmp.libhyper import NoConvergence
def _poly_from_expr(expr, opt):
    """Construct a polynomial from an expression. """
    orig, expr = (expr, sympify(expr))
    if not isinstance(expr, Basic):
        raise PolificationFailed(opt, orig, expr)
    elif expr.is_Poly:
        poly = expr.__class__._from_poly(expr, opt)
        opt.gens = poly.gens
        opt.domain = poly.domain
        if opt.polys is None:
            opt.polys = True
        return (poly, opt)
    elif opt.expand:
        expr = expr.expand()
    rep, opt = _dict_from_expr(expr, opt)
    if not opt.gens:
        raise PolificationFailed(opt, orig, expr)
    monoms, coeffs = list(zip(*list(rep.items())))
    domain = opt.domain
    if domain is None:
        opt.domain, coeffs = construct_domain(coeffs, opt=opt)
    else:
        coeffs = list(map(domain.from_sympy, coeffs))
    rep = dict(list(zip(monoms, coeffs)))
    poly = Poly._from_dict(rep, opt)
    if opt.polys is None:
        opt.polys = False
    return (poly, opt)