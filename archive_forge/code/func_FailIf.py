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
def FailIf(p, ctx=None):
    """Return a tactic that fails if the probe `p` evaluates to true.
    Otherwise, it returns the input goal unmodified.

    In the following example, the tactic applies 'simplify' if and only if there are
    more than 2 constraints in the goal.

    >>> t = OrElse(FailIf(Probe('size') > 2), Tactic('simplify'))
    >>> x, y = Ints('x y')
    >>> g = Goal()
    >>> g.add(x > 0)
    >>> g.add(y > 0)
    >>> t(g)
    [[x > 0, y > 0]]
    >>> g.add(x == y + 1)
    >>> t(g)
    [[Not(x <= 0), Not(y <= 0), x == 1 + y]]
    """
    p = _to_probe(p, ctx)
    return Tactic(Z3_tactic_fail_if(p.ctx.ref(), p.probe), p.ctx)