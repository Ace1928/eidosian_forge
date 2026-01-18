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
def _gcd_monom(f, g):
    ring = f.ring
    ground_gcd = ring.domain.gcd
    ground_quo = ring.domain.quo
    monomial_gcd = ring.monomial_gcd
    monomial_ldiv = ring.monomial_ldiv
    mf, cf = list(f.iterterms())[0]
    _mgcd, _cgcd = (mf, cf)
    for mg, cg in g.iterterms():
        _mgcd = monomial_gcd(_mgcd, mg)
        _cgcd = ground_gcd(_cgcd, cg)
    h = f.new([(_mgcd, _cgcd)])
    cff = f.new([(monomial_ldiv(mf, _mgcd), ground_quo(cf, _cgcd))])
    cfg = f.new([(monomial_ldiv(mg, _mgcd), ground_quo(cg, _cgcd)) for mg, cg in g.iterterms()])
    return (h, cff, cfg)