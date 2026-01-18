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
def Cond(p, t1, t2, ctx=None):
    """Return a tactic that applies tactic `t1` to a goal if probe `p` evaluates to true, and `t2` otherwise.

    >>> t = Cond(Probe('is-qfnra'), Tactic('qfnra'), Tactic('smt'))
    """
    p = _to_probe(p, ctx)
    t1 = _to_tactic(t1, ctx)
    t2 = _to_tactic(t2, ctx)
    return Tactic(Z3_tactic_cond(t1.ctx.ref(), p.probe, t1.tactic, t2.tactic), t1.ctx)