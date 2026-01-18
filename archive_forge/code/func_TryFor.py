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
def TryFor(t, ms, ctx=None):
    """Return a tactic that applies `t` to a given goal for `ms` milliseconds.

    If `t` does not terminate in `ms` milliseconds, then it fails.
    """
    t = _to_tactic(t, ctx)
    return Tactic(Z3_tactic_try_for(t.ctx.ref(), t.tactic, ms), t.ctx)