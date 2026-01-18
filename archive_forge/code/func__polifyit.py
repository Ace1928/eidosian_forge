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
def _polifyit(func):

    @wraps(func)
    def wrapper(f, g):
        g = _sympify(g)
        if isinstance(g, Poly):
            return func(f, g)
        elif isinstance(g, Expr):
            try:
                g = f.from_expr(g, *f.gens)
            except PolynomialError:
                if g.is_Matrix:
                    return NotImplemented
                expr_method = getattr(f.as_expr(), func.__name__)
                result = expr_method(g)
                if result is not NotImplemented:
                    sympy_deprecation_warning('\n                        Mixing Poly with non-polynomial expressions in binary\n                        operations is deprecated. Either explicitly convert\n                        the non-Poly operand to a Poly with as_poly() or\n                        convert the Poly to an Expr with as_expr().\n                        ', deprecated_since_version='1.6', active_deprecations_target='deprecated-poly-nonpoly-binary-operations')
                return result
            else:
                return func(f, g)
        else:
            return NotImplemented
    return wrapper