from sympy.core.sympify import sympify
from sympy.core import (S, Pow, Dummy, pi, Expr, Wild, Mul, Equality,
from sympy.core.containers import Tuple
from sympy.core.function import (Lambda, expand_complex, AppliedUndef,
from sympy.core.mod import Mod
from sympy.core.numbers import igcd, I, Number, Rational, oo, ilcm
from sympy.core.power import integer_log
from sympy.core.relational import Eq, Ne, Relational
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Symbol, _uniquely_named_symbol
from sympy.core.sympify import _sympify
from sympy.polys.matrices.linsolve import _linear_eq_to_dict
from sympy.polys.polyroots import UnsolvableFactorError
from sympy.simplify.simplify import simplify, fraction, trigsimp, nsimplify
from sympy.simplify import powdenest, logcombine
from sympy.functions import (log, tan, cot, sin, cos, sec, csc, exp,
from sympy.functions.elementary.complexes import Abs, arg, re, im
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.miscellaneous import real_root
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.logic.boolalg import And, BooleanTrue
from sympy.sets import (FiniteSet, imageset, Interval, Intersection,
from sympy.sets.sets import Set, ProductSet
from sympy.matrices import zeros, Matrix, MatrixBase
from sympy.ntheory import totient
from sympy.ntheory.factor_ import divisors
from sympy.ntheory.residue_ntheory import discrete_log, nthroot_mod
from sympy.polys import (roots, Poly, degree, together, PolynomialError,
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polytools import invert, groebner, poly
from sympy.polys.solvers import (sympy_eqs_to_ring, solve_lin_sys,
from sympy.polys.matrices.linsolve import _linsolve
from sympy.solvers.solvers import (checksol, denoms, unrad,
from sympy.solvers.polysys import solve_poly_system
from sympy.utilities import filldedent
from sympy.utilities.iterables import (numbered_symbols, has_dups,
from sympy.calculus.util import periodicity, continuous_domain, function_range
from types import GeneratorType
def _solve_as_poly(f, symbol, domain=S.Complexes):
    """
    Solve the equation using polynomial techniques if it already is a
    polynomial equation or, with a change of variables, can be made so.
    """
    result = None
    if f.is_polynomial(symbol):
        solns = roots(f, symbol, cubics=True, quartics=True, quintics=True, domain='EX')
        num_roots = sum(solns.values())
        if degree(f, symbol) <= num_roots:
            result = FiniteSet(*solns.keys())
        else:
            poly = Poly(f, symbol)
            solns = poly.all_roots()
            if poly.degree() <= len(solns):
                result = FiniteSet(*solns)
            else:
                result = ConditionSet(symbol, Eq(f, 0), domain)
    else:
        poly = Poly(f)
        if poly is None:
            result = ConditionSet(symbol, Eq(f, 0), domain)
        gens = [g for g in poly.gens if g.has(symbol)]
        if len(gens) == 1:
            poly = Poly(poly, gens[0])
            gen = poly.gen
            deg = poly.degree()
            poly = Poly(poly.as_expr(), poly.gen, composite=True)
            poly_solns = FiniteSet(*roots(poly, cubics=True, quartics=True, quintics=True).keys())
            if len(poly_solns) < deg:
                result = ConditionSet(symbol, Eq(f, 0), domain)
            if gen != symbol:
                y = Dummy('y')
                inverter = invert_real if domain.is_subset(S.Reals) else invert_complex
                lhs, rhs_s = inverter(gen, y, symbol)
                if lhs == symbol:
                    result = Union(*[rhs_s.subs(y, s) for s in poly_solns])
                    if isinstance(result, FiniteSet) and isinstance(gen, Pow) and gen.base.is_Rational:
                        result = FiniteSet(*[expand_log(i) for i in result])
                else:
                    result = ConditionSet(symbol, Eq(f, 0), domain)
        else:
            result = ConditionSet(symbol, Eq(f, 0), domain)
    if result is not None:
        if isinstance(result, FiniteSet):
            if all((s.atoms(Symbol, AppliedUndef) == set() and (not isinstance(s, RootOf)) for s in result)):
                s = Dummy('s')
                result = imageset(Lambda(s, expand_complex(s)), result)
        if isinstance(result, FiniteSet) and domain != S.Complexes:
            result = result.intersection(domain)
        return result
    else:
        return ConditionSet(symbol, Eq(f, 0), domain)