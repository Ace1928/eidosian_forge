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
def _extract_main_soln(sym, sol, soln_imageset):
    """Separate the Complements, Intersections, ImageSet lambda expr and
        its base_set. This function returns the unmasks sol from different classes
        of sets and also returns the appended ImageSet elements in a
        soln_imageset (dict: where key as unmasked element and value as ImageSet).
        """
    if isinstance(sol, ConditionSet):
        sol = sol.base_set
    if isinstance(sol, Complement):
        complements[sym] = sol.args[1]
        sol = sol.args[0]
    if isinstance(sol, Union):
        sol_args = sol.args
        sol = S.EmptySet
        for sol_arg2 in sol_args:
            if isinstance(sol_arg2, FiniteSet):
                sol += sol_arg2
            else:
                sol += FiniteSet(sol_arg2)
    if isinstance(sol, Intersection):
        if sol.args[0] not in (S.Reals, S.Complexes):
            intersections[sym] = sol.args[0]
        sol = sol.args[1]
    if isinstance(sol, ImageSet):
        soln_imagest = sol
        expr2 = sol.lamda.expr
        sol = FiniteSet(expr2)
        soln_imageset[expr2] = soln_imagest
    if not isinstance(sol, FiniteSet):
        sol = FiniteSet(sol)
    return (sol, soln_imageset)