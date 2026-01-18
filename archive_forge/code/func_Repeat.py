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
def Repeat(t, max=4294967295, ctx=None):
    """Return a tactic that keeps applying `t` until the goal is not modified anymore
    or the maximum number of iterations `max` is reached.

    >>> x, y = Ints('x y')
    >>> c = And(Or(x == 0, x == 1), Or(y == 0, y == 1), x > y)
    >>> t = Repeat(OrElse(Tactic('split-clause'), Tactic('skip')))
    >>> r = t(c)
    >>> for subgoal in r: print(subgoal)
    [x == 0, y == 0, x > y]
    [x == 0, y == 1, x > y]
    [x == 1, y == 0, x > y]
    [x == 1, y == 1, x > y]
    >>> t = Then(t, Tactic('propagate-values'))
    >>> t(c)
    [[x == 1, y == 0]]
    """
    t = _to_tactic(t, ctx)
    return Tactic(Z3_tactic_repeat(t.ctx.ref(), t.tactic, max), t.ctx)