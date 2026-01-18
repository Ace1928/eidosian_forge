from __future__ import annotations
from typing import Any
from operator import add, mul, lt, le, gt, ge
from functools import reduce
from types import GeneratorType
from sympy.core.expr import Expr
from sympy.core.numbers import igcd, oo
from sympy.core.symbol import Symbol, symbols as _symbols
from sympy.core.sympify import CantSympify, sympify
from sympy.ntheory.multinomial import multinomial_coefficients
from sympy.polys.compatibility import IPolys
from sympy.polys.constructor import construct_domain
from sympy.polys.densebasic import dmp_to_dict, dmp_from_dict
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.polynomialring import PolynomialRing
from sympy.polys.heuristicgcd import heugcd
from sympy.polys.monomials import MonomialOps
from sympy.polys.orderings import lex
from sympy.polys.polyerrors import (
from sympy.polys.polyoptions import (Domain as DomainOpt,
from sympy.polys.polyutils import (expr_from_dict, _dict_reorder,
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public, subsets
from sympy.utilities.iterables import is_sequence
from sympy.utilities.magic import pollute
def _iadd_poly_monom(self, p2, mc):
    """add to self the product of (p)*(coeff*x0**i0*x1**i1*...)
        unless self is a generator -- then just return the sum of the two.

        mc is a tuple, (monom, coeff), where monomial is (i0, i1, ...)

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y, z = ring('x, y, z', ZZ)
        >>> p1 = x**4 + 2*y
        >>> p2 = y + z
        >>> m = (1, 2, 3)
        >>> p1 = p1._iadd_poly_monom(p2, (m, 3))
        >>> p1
        x**4 + 3*x*y**3*z**3 + 3*x*y**2*z**4 + 2*y

        """
    p1 = self
    if p1 in p1.ring._gens_set:
        p1 = p1.copy()
    m, c = mc
    get = p1.get
    zero = p1.ring.domain.zero
    monomial_mul = p1.ring.monomial_mul
    for k, v in p2.items():
        ka = monomial_mul(k, m)
        coeff = get(ka, zero) + v * c
        if coeff:
            p1[ka] = coeff
        else:
            del p1[ka]
    return p1