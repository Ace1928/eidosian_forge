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
def _try_rescale(f, f1=None):
    """
        try rescaling ``x -> alpha*x`` to convert f to a polynomial
        with rational coefficients.
        Returns ``alpha, f``; if the rescaling is successful,
        ``alpha`` is the rescaling factor, and ``f`` is the rescaled
        polynomial; else ``alpha`` is ``None``.
        """
    if not len(f.gens) == 1 or not f.gens[0].is_Atom:
        return (None, f)
    n = f.degree()
    lc = f.LC()
    f1 = f1 or f1.monic()
    coeffs = f1.all_coeffs()[1:]
    coeffs = [simplify(coeffx) for coeffx in coeffs]
    if len(coeffs) > 1 and coeffs[-2]:
        rescale1_x = simplify(coeffs[-2] / coeffs[-1])
        coeffs1 = []
        for i in range(len(coeffs)):
            coeffx = simplify(coeffs[i] * rescale1_x ** (i + 1))
            if not coeffx.is_rational:
                break
            coeffs1.append(coeffx)
        else:
            rescale_x = simplify(1 / rescale1_x)
            x = f.gens[0]
            v = [x ** n]
            for i in range(1, n + 1):
                v.append(coeffs1[i - 1] * x ** (n - i))
            f = Add(*v)
            f = Poly(f)
            return (lc, rescale_x, f)
    return None