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
def _solve_exponential(lhs, rhs, symbol, domain):
    """
    Helper function for solving (supported) exponential equations.

    Exponential equations are the sum of (currently) at most
    two terms with one or both of them having a power with a
    symbol-dependent exponent.

    For example

    .. math:: 5^{2x + 3} - 5^{3x - 1}

    .. math:: 4^{5 - 9x} - e^{2 - x}

    Parameters
    ==========

    lhs, rhs : Expr
        The exponential equation to be solved, `lhs = rhs`

    symbol : Symbol
        The variable in which the equation is solved

    domain : Set
        A set over which the equation is solved.

    Returns
    =======

    A set of solutions satisfying the given equation.
    A ``ConditionSet`` if the equation is unsolvable or
    if the assumptions are not properly defined, in that case
    a different style of ``ConditionSet`` is returned having the
    solution(s) of the equation with the desired assumptions.

    Examples
    ========

    >>> from sympy.solvers.solveset import _solve_exponential as solve_expo
    >>> from sympy import symbols, S
    >>> x = symbols('x', real=True)
    >>> a, b = symbols('a b')
    >>> solve_expo(2**x + 3**x - 5**x, 0, x, S.Reals)  # not solvable
    ConditionSet(x, Eq(2**x + 3**x - 5**x, 0), Reals)
    >>> solve_expo(a**x - b**x, 0, x, S.Reals)  # solvable but incorrect assumptions
    ConditionSet(x, (a > 0) & (b > 0), {0})
    >>> solve_expo(3**(2*x) - 2**(x + 3), 0, x, S.Reals)
    {-3*log(2)/(-2*log(3) + log(2))}
    >>> solve_expo(2**x - 4**x, 0, x, S.Reals)
    {0}

    * Proof of correctness of the method

    The logarithm function is the inverse of the exponential function.
    The defining relation between exponentiation and logarithm is:

    .. math:: {\\log_b x} = y \\enspace if \\enspace b^y = x

    Therefore if we are given an equation with exponent terms, we can
    convert every term to its corresponding logarithmic form. This is
    achieved by taking logarithms and expanding the equation using
    logarithmic identities so that it can easily be handled by ``solveset``.

    For example:

    .. math:: 3^{2x} = 2^{x + 3}

    Taking log both sides will reduce the equation to

    .. math:: (2x)\\log(3) = (x + 3)\\log(2)

    This form can be easily handed by ``solveset``.
    """
    unsolved_result = ConditionSet(symbol, Eq(lhs - rhs, 0), domain)
    newlhs = powdenest(lhs)
    if lhs != newlhs:
        neweq = factor(newlhs - rhs)
        if neweq != lhs - rhs:
            return _solveset(neweq, symbol, domain)
    if not (isinstance(lhs, Add) and len(lhs.args) == 2):
        return unsolved_result
    if rhs != 0:
        return unsolved_result
    a, b = list(ordered(lhs.args))
    a_term = a.as_independent(symbol)[1]
    b_term = b.as_independent(symbol)[1]
    a_base, a_exp = a_term.as_base_exp()
    b_base, b_exp = b_term.as_base_exp()
    if domain.is_subset(S.Reals):
        conditions = And(a_base > 0, b_base > 0, Eq(im(a_exp), 0), Eq(im(b_exp), 0))
    else:
        conditions = And(Ne(a_base, 0), Ne(b_base, 0))
    L, R = (expand_log(log(i), force=True) for i in (a, -b))
    solutions = _solveset(L - R, symbol, domain)
    return ConditionSet(symbol, conditions, solutions)