from collections import OrderedDict
from functools import reduce
import math
from operator import add
from ..units import get_derived_unit, default_units, energy, concentration
from ..util._dimensionality import dimension_codes, base_registry
from ..util.pyutil import memoize, deprecated
from ..util._expr import Expr, UnaryWrapper, Symbol
class RampedTemp(Expr):
    """Ramped temperature, pass as substitution to e.g. ``get_odesys``"""
    argument_names = ('T0', 'dTdt')
    parameter_keys = ('time',)

    def args_dimensionality(self, **kwargs):
        return ({'temperature': 1}, {'temperature': 1, 'time': -1})

    def __call__(self, variables, backend=None, **kwargs):
        T0, dTdt = self.all_args(variables, backend=backend, **kwargs)
        return T0 + dTdt * variables['time']