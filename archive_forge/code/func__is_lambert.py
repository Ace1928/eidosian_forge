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
def _is_lambert(f, symbol):
    """
    If this returns ``False`` then the Lambert solver (``_solve_lambert``) will not be called.

    Explanation
    ===========

    Quick check for cases that the Lambert solver might be able to handle.

    1. Equations containing more than two operands and `symbol`s involving any of
       `Pow`, `exp`, `HyperbolicFunction`,`TrigonometricFunction`, `log` terms.

    2. In `Pow`, `exp` the exponent should have `symbol` whereas for
       `HyperbolicFunction`,`TrigonometricFunction`, `log` should contain `symbol`.

    3. For `HyperbolicFunction`,`TrigonometricFunction` the number of trigonometric functions in
       equation should be less than number of symbols. (since `A*cos(x) + B*sin(x) - c`
       is not the Lambert type).

    Some forms of lambert equations are:
        1. X**X = C
        2. X*(B*log(X) + D)**A = C
        3. A*log(B*X + A) + d*X = C
        4. (B*X + A)*exp(d*X + g) = C
        5. g*exp(B*X + h) - B*X = C
        6. A*D**(E*X + g) - B*X = C
        7. A*cos(X) + B*sin(X) - D*X = C
        8. A*cosh(X) + B*sinh(X) - D*X = C

    Where X is any variable,
          A, B, C, D, E are any constants,
          g, h are linear functions or log terms.

    Parameters
    ==========

    f : Expr
        The equation to be checked

    symbol : Symbol
        The variable in which the equation is checked

    Returns
    =======

    If this returns ``False`` then the Lambert solver (``_solve_lambert``) will not be called.

    Examples
    ========

    >>> from sympy.solvers.solveset import _is_lambert
    >>> from sympy import symbols, cosh, sinh, log
    >>> x = symbols('x')

    >>> _is_lambert(3*log(x) - x*log(3), x)
    True
    >>> _is_lambert(log(log(x - 3)) + log(x-3), x)
    True
    >>> _is_lambert(cosh(x) - sinh(x), x)
    False
    >>> _is_lambert((x**2 - 2*x + 1).subs(x, (log(x) + 3*x)**2 - 1), x)
    True

    See Also
    ========

    _solve_lambert

    """
    term_factors = list(_term_factors(f.expand()))
    no_of_symbols = len([arg for arg in term_factors if arg.has(symbol)])
    no_of_trig = len([arg for arg in term_factors if arg.has(HyperbolicFunction, TrigonometricFunction)])
    if f.is_Add and no_of_symbols >= 2:
        lambert_funcs = (log, HyperbolicFunction, TrigonometricFunction)
        if any((isinstance(arg, lambert_funcs) for arg in term_factors if arg.has(symbol))):
            if no_of_trig < no_of_symbols:
                return True
        elif any((isinstance(arg, (Pow, exp)) for arg in term_factors if arg.as_base_exp()[1].has(symbol))):
            return True
    return False