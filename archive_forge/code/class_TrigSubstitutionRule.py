from __future__ import annotations
from typing import NamedTuple, Type, Callable, Sequence
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict
from collections.abc import Mapping
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.containers import Dict
from sympy.core.expr import Expr
from sympy.core.function import Derivative
from sympy.core.logic import fuzzy_not
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, Number, E
from sympy.core.power import Pow
from sympy.core.relational import Eq, Ne, Boolean
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol, Wild
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.hyperbolic import (HyperbolicFunction, csch,
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (TrigonometricFunction,
from sympy.functions.special.delta_functions import Heaviside, DiracDelta
from sympy.functions.special.error_functions import (erf, erfi, fresnelc,
from sympy.functions.special.gamma_functions import uppergamma
from sympy.functions.special.elliptic_integrals import elliptic_e, elliptic_f
from sympy.functions.special.polynomials import (chebyshevt, chebyshevu,
from sympy.functions.special.zeta_functions import polylog
from .integrals import Integral
from sympy.logic.boolalg import And
from sympy.ntheory.factor_ import primefactors
from sympy.polys.polytools import degree, lcm_list, gcd_list, Poly
from sympy.simplify.radsimp import fraction
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve
from sympy.strategies.core import switch, do_one, null_safe, condition
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import debug
@dataclass
class TrigSubstitutionRule(Rule):
    theta: Expr
    func: Expr
    rewritten: Expr
    substep: Rule
    restriction: bool | Boolean

    def eval(self) -> Expr:
        theta, func, x = (self.theta, self.func, self.variable)
        func = func.subs(sec(theta), 1 / cos(theta))
        func = func.subs(csc(theta), 1 / sin(theta))
        func = func.subs(cot(theta), 1 / tan(theta))
        trig_function = list(func.find(TrigonometricFunction))
        assert len(trig_function) == 1
        trig_function = trig_function[0]
        relation = solve(x - func, trig_function)
        assert len(relation) == 1
        numer, denom = fraction(relation[0])
        if isinstance(trig_function, sin):
            opposite = numer
            hypotenuse = denom
            adjacent = sqrt(denom ** 2 - numer ** 2)
            inverse = asin(relation[0])
        elif isinstance(trig_function, cos):
            adjacent = numer
            hypotenuse = denom
            opposite = sqrt(denom ** 2 - numer ** 2)
            inverse = acos(relation[0])
        else:
            opposite = numer
            adjacent = denom
            hypotenuse = sqrt(denom ** 2 + numer ** 2)
            inverse = atan(relation[0])
        substitution = [(sin(theta), opposite / hypotenuse), (cos(theta), adjacent / hypotenuse), (tan(theta), opposite / adjacent), (theta, inverse)]
        return Piecewise((self.substep.eval().subs(substitution).trigsimp(), self.restriction))

    def contains_dont_know(self) -> bool:
        return self.substep.contains_dont_know()