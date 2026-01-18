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
def _iadd_monom(self, mc):
    """add to self the monomial coeff*x0**i0*x1**i1*...
        unless self is a generator -- then just return the sum of the two.

        mc is a tuple, (monom, coeff), where monomial is (i0, i1, ...)

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y = ring('x, y', ZZ)
        >>> p = x**4 + 2*y
        >>> m = (1, 2)
        >>> p1 = p._iadd_monom((m, 5))
        >>> p1
        x**4 + 5*x*y**2 + 2*y
        >>> p1 is p
        True
        >>> p = x
        >>> p1 = p._iadd_monom((m, 5))
        >>> p1
        5*x*y**2 + x
        >>> p1 is p
        False

        """
    if self in self.ring._gens_set:
        cpself = self.copy()
    else:
        cpself = self
    expv, coeff = mc
    c = cpself.get(expv)
    if c is None:
        cpself[expv] = coeff
    else:
        c += coeff
        if c:
            cpself[expv] = c
        else:
            del cpself[expv]
    return cpself