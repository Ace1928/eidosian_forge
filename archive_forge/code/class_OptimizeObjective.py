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
class OptimizeObjective:

    def __init__(self, opt, value, is_max):
        self._opt = opt
        self._value = value
        self._is_max = is_max

    def lower(self):
        opt = self._opt
        return _to_expr_ref(Z3_optimize_get_lower(opt.ctx.ref(), opt.optimize, self._value), opt.ctx)

    def upper(self):
        opt = self._opt
        return _to_expr_ref(Z3_optimize_get_upper(opt.ctx.ref(), opt.optimize, self._value), opt.ctx)

    def lower_values(self):
        opt = self._opt
        return AstVector(Z3_optimize_get_lower_as_vector(opt.ctx.ref(), opt.optimize, self._value), opt.ctx)

    def upper_values(self):
        opt = self._opt
        return AstVector(Z3_optimize_get_upper_as_vector(opt.ctx.ref(), opt.optimize, self._value), opt.ctx)

    def value(self):
        if self._is_max:
            return self.upper()
        else:
            return self.lower()

    def __str__(self):
        return '%s:%s' % (self._value, self._is_max)