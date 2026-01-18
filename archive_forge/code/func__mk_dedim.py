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
def _mk_dedim(unit_registry):
    unit_time = get_derived_unit(unit_registry, 'time')
    unit_conc = get_derived_unit(unit_registry, 'concentration')

    def dedim_tcp(t, c, p, param_unit=lambda k, v: default_unit_in_registry(v, unit_registry)):
        _t = to_unitless(t, unit_time)
        _c = to_unitless(c, unit_conc)
        _p, pu = ({}, {})
        for k, v in p.items():
            pu[k] = param_unit(k, v)
            _p[k] = to_unitless(v, pu[k])
        return ((_t, _c, _p), dict(unit_time=unit_time, unit_conc=unit_conc, param_units=pu))
    return locals()