from sympy.core import S, Pow
from sympy.core.function import (Derivative, AppliedUndef, diff)
from sympy.core.relational import Equality, Eq
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.logic.boolalg import BooleanAtom
from sympy.functions import exp
from sympy.series import Order
from sympy.simplify.simplify import simplify, posify, besselsimp
from sympy.simplify.trigsimp import trigsimp
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.solvers import solve
from sympy.solvers.deutils import _preprocess, ode_order
from sympy.utilities.iterables import iterable, is_sequence
def _sympify(eq):
    return list(map(sympify, eq if iterable(eq) else [eq]))