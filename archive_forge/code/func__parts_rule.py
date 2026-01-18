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
def _parts_rule(integrand, symbol) -> tuple[Expr, Expr, Expr, Expr, Rule] | None:

    def pull_out_algebraic(integrand):
        integrand = integrand.cancel().together()
        algebraic = [] if isinstance(integrand, Piecewise) or not integrand.is_Mul else [arg for arg in integrand.args if arg.is_algebraic_expr(symbol)]
        if algebraic:
            u = Mul(*algebraic)
            dv = (integrand / u).cancel()
            return (u, dv)

    def pull_out_u(*functions) -> Callable[[Expr], tuple[Expr, Expr] | None]:

        def pull_out_u_rl(integrand: Expr) -> tuple[Expr, Expr] | None:
            if any((integrand.has(f) for f in functions)):
                args = [arg for arg in integrand.args if any((isinstance(arg, cls) for cls in functions))]
                if args:
                    u = Mul(*args)
                    dv = integrand / u
                    return (u, dv)
            return None
        return pull_out_u_rl
    liate_rules = [pull_out_u(log), pull_out_u(*inverse_trig_functions), pull_out_algebraic, pull_out_u(sin, cos), pull_out_u(exp)]
    dummy = Dummy('temporary')
    if isinstance(integrand, (log, *inverse_trig_functions)):
        integrand = dummy * integrand
    for index, rule in enumerate(liate_rules):
        result = rule(integrand)
        if result:
            u, dv = result
            if symbol not in u.free_symbols and (not u.has(dummy)):
                return None
            u = u.subs(dummy, 1)
            dv = dv.subs(dummy, 1)
            if rule == pull_out_algebraic and (not u.is_polynomial(symbol)):
                return None
            if isinstance(u, log):
                rec_dv = 1 / dv
                if rec_dv.is_polynomial(symbol) and degree(rec_dv, symbol) == 1:
                    return None
            if rule == pull_out_algebraic:
                if dv.is_Derivative or dv.has(TrigonometricFunction) or isinstance(dv, OrthogonalPolynomial):
                    v_step = integral_steps(dv, symbol)
                    if v_step.contains_dont_know():
                        return None
                    else:
                        du = u.diff(symbol)
                        v = v_step.eval()
                        return (u, dv, v, du, v_step)
            accept = False
            if index < 2:
                accept = True
            elif rule == pull_out_algebraic and dv.args and all((isinstance(a, (sin, cos, exp)) for a in dv.args)):
                accept = True
            else:
                for lrule in liate_rules[index + 1:]:
                    r = lrule(integrand)
                    if r and r[0].subs(dummy, 1).equals(dv):
                        accept = True
                        break
            if accept:
                du = u.diff(symbol)
                v_step = integral_steps(simplify(dv), symbol)
                if not v_step.contains_dont_know():
                    v = v_step.eval()
                    return (u, dv, v, du, v_step)
    return None