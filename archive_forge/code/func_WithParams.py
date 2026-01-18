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
def WithParams(t, p):
    """Return a tactic that applies tactic `t` using the given configuration options.

    >>> x, y = Ints('x y')
    >>> p = ParamsRef()
    >>> p.set("som", True)
    >>> t = WithParams(Tactic('simplify'), p)
    >>> t((x + 1)*(y + 2) == 0)
    [[2*x + y + x*y == -2]]
    """
    t = _to_tactic(t, None)
    return Tactic(Z3_tactic_using_params(t.ctx.ref(), t.tactic, p.params), t.ctx)