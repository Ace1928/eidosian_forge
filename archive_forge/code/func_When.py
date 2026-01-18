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
def When(p, t, ctx=None):
    """Return a tactic that applies tactic `t` only if probe `p` evaluates to true.
    Otherwise, it returns the input goal unmodified.

    >>> t = When(Probe('size') > 2, Tactic('simplify'))
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
    t = _to_tactic(t, ctx)
    return Tactic(Z3_tactic_when(t.ctx.ref(), p.probe, t.tactic), t.ctx)