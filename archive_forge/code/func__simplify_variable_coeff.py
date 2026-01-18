from functools import reduce
from itertools import combinations_with_replacement
from sympy.simplify import simplify  # type: ignore
from sympy.core import Add, S
from sympy.core.function import Function, expand, AppliedUndef, Subs
from sympy.core.relational import Equality, Eq
from sympy.core.symbol import Symbol, Wild, symbols
from sympy.functions import exp
from sympy.integrals.integrals import Integral, integrate
from sympy.utilities.iterables import has_dups, is_sequence
from sympy.utilities.misc import filldedent
from sympy.solvers.deutils import _preprocess, ode_order, _desolve
from sympy.solvers.solvers import solve
from sympy.simplify.radsimp import collect
import operator
def _simplify_variable_coeff(sol, syms, func, funcarg):
    """
    Helper function to replace constants by functions in 1st_linear_variable_coeff
    """
    eta = Symbol('eta')
    if len(syms) == 1:
        sym = syms.pop()
        final = sol.subs(sym, func(funcarg))
    else:
        for key, sym in enumerate(syms):
            final = sol.subs(sym, func(funcarg))
    return simplify(final.subs(eta, funcarg))