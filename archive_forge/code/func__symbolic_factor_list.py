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
def _symbolic_factor_list(expr, opt, method):
    """Helper function for :func:`_symbolic_factor`. """
    coeff, factors = (S.One, [])
    args = [i._eval_factor() if hasattr(i, '_eval_factor') else i for i in Mul.make_args(expr)]
    for arg in args:
        if arg.is_Number or (isinstance(arg, Expr) and pure_complex(arg)):
            coeff *= arg
            continue
        elif arg.is_Pow and arg.base != S.Exp1:
            base, exp = arg.args
            if base.is_Number and exp.is_Number:
                coeff *= arg
                continue
            if base.is_Number:
                factors.append((base, exp))
                continue
        else:
            base, exp = (arg, S.One)
        try:
            poly, _ = _poly_from_expr(base, opt)
        except PolificationFailed as exc:
            factors.append((exc.expr, exp))
        else:
            func = getattr(poly, method + '_list')
            _coeff, _factors = func()
            if _coeff is not S.One:
                if exp.is_Integer:
                    coeff *= _coeff ** exp
                elif _coeff.is_positive:
                    factors.append((_coeff, exp))
                else:
                    _factors.append((_coeff, S.One))
            if exp is S.One:
                factors.extend(_factors)
            elif exp.is_integer:
                factors.extend([(f, k * exp) for f, k in _factors])
            else:
                other = []
                for f, k in _factors:
                    if f.as_expr().is_positive:
                        factors.append((f, k * exp))
                    else:
                        other.append((f, k))
                factors.append((_factors_product(other), exp))
    if method == 'sqf':
        factors = [(reduce(mul, (f for f, _ in factors if _ == k)), k) for k in {i for _, i in factors}]
    return (coeff, factors)