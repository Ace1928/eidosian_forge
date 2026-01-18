from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
def SolverFor(logic, ctx=None, logFile=None):
    """Create a solver customized for the given logic.

    The parameter `logic` is a string. It should be contains
    the name of a SMT-LIB logic.
    See http://www.smtlib.org/ for the name of all available logics.

    >>> s = SolverFor("QF_LIA")
    >>> x = Int('x')
    >>> s.add(x > 0)
    >>> s.add(x < 2)
    >>> s.check()
    sat
    >>> s.model()
    [x = 1]
    """
    ctx = _get_ctx(ctx)
    logic = to_symbol(logic)
    return Solver(Z3_mk_solver_for_logic(ctx.ref(), logic), ctx, logFile)