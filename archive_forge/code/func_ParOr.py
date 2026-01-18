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
def ParOr(*ts, **ks):
    """Return a tactic that applies the tactics in `*ts` in parallel until one of them succeeds (it doesn't fail).

    >>> x = Int('x')
    >>> t = ParOr(Tactic('simplify'), Tactic('fail'))
    >>> t(x + 1 == 2)
    [[x == 1]]
    """
    if z3_debug():
        _z3_assert(len(ts) >= 2, 'At least two arguments expected')
    ctx = _get_ctx(ks.get('ctx', None))
    ts = [_to_tactic(t, ctx) for t in ts]
    sz = len(ts)
    _args = (TacticObj * sz)()
    for i in range(sz):
        _args[i] = ts[i].tactic
    return Tactic(Z3_tactic_par_or(ctx.ref(), sz, _args), ctx)