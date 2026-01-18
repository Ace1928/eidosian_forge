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
def _solve_trig1(f, symbol, domain):
    """Primary solver for trigonometric and hyperbolic equations

    Returns either the solution set as a ConditionSet (auto-evaluated to a
    union of ImageSets if no variables besides 'symbol' are involved) or
    raises _SolveTrig1Error if f == 0 cannot be solved.

    Notes
    =====
    Algorithm:
    1. Do a change of variable x -> mu*x in arguments to trigonometric and
    hyperbolic functions, in order to reduce them to small integers. (This
    step is crucial to keep the degrees of the polynomials of step 4 low.)
    2. Rewrite trigonometric/hyperbolic functions as exponentials.
    3. Proceed to a 2nd change of variable, replacing exp(I*x) or exp(x) by y.
    4. Solve the resulting rational equation.
    5. Use invert_complex or invert_real to return to the original variable.
    6. If the coefficients of 'symbol' were symbolic in nature, add the
    necessary consistency conditions in a ConditionSet.

    """
    x = Dummy('x')
    if _is_function_class_equation(HyperbolicFunction, f, symbol):
        cov = exp(x)
        inverter = invert_real if domain.is_subset(S.Reals) else invert_complex
    else:
        cov = exp(I * x)
        inverter = invert_complex
    f = trigsimp(f)
    f_original = f
    trig_functions = f.atoms(TrigonometricFunction, HyperbolicFunction)
    trig_arguments = [e.args[0] for e in trig_functions]
    if not any((a.has(symbol) for a in trig_arguments)):
        return solveset(f_original, symbol, domain)
    denominators = []
    numerators = []
    for ar in trig_arguments:
        try:
            poly_ar = Poly(ar, symbol)
        except PolynomialError:
            raise _SolveTrig1Error('trig argument is not a polynomial')
        if poly_ar.degree() > 1:
            raise _SolveTrig1Error('degree of variable must not exceed one')
        if poly_ar.degree() == 0:
            continue
        c = poly_ar.all_coeffs()[0]
        numerators.append(fraction(c)[0])
        denominators.append(fraction(c)[1])
    mu = lcm(denominators) / gcd(numerators)
    f = f.subs(symbol, mu * x)
    f = f.rewrite(exp)
    f = together(f)
    g, h = fraction(f)
    y = Dummy('y')
    g, h = (g.expand(), h.expand())
    g, h = (g.subs(cov, y), h.subs(cov, y))
    if g.has(x) or h.has(x):
        raise _SolveTrig1Error('change of variable not possible')
    solns = solveset_complex(g, y) - solveset_complex(h, y)
    if isinstance(solns, ConditionSet):
        raise _SolveTrig1Error('polynomial has ConditionSet solution')
    if isinstance(solns, FiniteSet):
        if any((isinstance(s, RootOf) for s in solns)):
            raise _SolveTrig1Error('polynomial results in RootOf object')
        cov = cov.subs(x, symbol / mu)
        result = Union(*[inverter(cov, s, symbol)[1] for s in solns])
        if mu.has(Symbol):
            syms = mu.atoms(Symbol)
            munum, muden = fraction(mu)
            condnum = munum.as_independent(*syms, as_Add=False)[1]
            condden = muden.as_independent(*syms, as_Add=False)[1]
            cond = And(Ne(condnum, 0), Ne(condden, 0))
        else:
            cond = True
        if domain is S.Complexes:
            return ConditionSet(symbol, cond, result)
        else:
            return ConditionSet(symbol, cond, Intersection(result, domain))
    elif solns is S.EmptySet:
        return S.EmptySet
    else:
        raise _SolveTrig1Error('polynomial solutions must form FiniteSet')