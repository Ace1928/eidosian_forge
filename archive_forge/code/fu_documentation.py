from collections import defaultdict
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.exprtools import Factors, gcd_terms, factor_terms
from sympy.core.function import expand_mul
from sympy.core.mul import Mul
from sympy.core.numbers import pi, I
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.core.traversal import bottom_up
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.hyperbolic import (
from sympy.functions.elementary.trigonometric import (
from sympy.ntheory.factor_ import perfect_power
from sympy.polys.polytools import factor
from sympy.strategies.tree import greedy
from sympy.strategies.core import identity, debug
from sympy import SYMPY_DEBUG
Return ``a`` as a tuple (r, c, s) such that
        ``a = (r or 1)*(c or 1)*(s or 1)``.

        Three arguments are returned (radical, c-factor, s-factor) as
        long as the conditions set by ``two`` are met; otherwise None is
        returned. If ``two`` is True there will be one or two non-None
        values in the tuple: c and s or c and r or s and r or s or c with c
        being a cosine function (if possible) else a sine, and s being a sine
        function (if possible) else oosine. If ``two`` is False then there
        will only be a c or s term in the tuple.

        ``two`` also require that either two cos and/or sin be present (with
        the condition that if the functions are the same the arguments are
        different or vice versa) or that a single cosine or a single sine
        be present with an optional radical.

        If the above conditions dictated by ``two`` are not met then None
        is returned.
        