from collections import OrderedDict
from functools import reduce, partial
from itertools import chain
from operator import attrgetter, mul
import math
import warnings
from ..units import (
from ..util.pyutil import deprecated
from ..util._expr import Expr, Symbol
from .rates import RateExpr, MassAction
def _mk_unit_aware_solve(odesys, unit_registry, validate):
    dedim_ctx = _mk_dedim(unit_registry)

    def solve(t, c, p, **kwargs):
        for name in odesys.names:
            c[name]
        validate(dict(c, **p))
        tcp, dedim_extra = dedim_ctx['dedim_tcp'](t, c, p)
        result = odesys.integrate(*tcp, **kwargs)
        result.xout = result.xout * dedim_extra['unit_time']
        result.yout = result.yout * dedim_extra['unit_conc']
        return (result, dedim_extra)
    return solve