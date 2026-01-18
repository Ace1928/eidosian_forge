from collections import OrderedDict
from functools import reduce
import math
from operator import add
from ..units import get_derived_unit, default_units, energy, concentration
from ..util._dimensionality import dimension_codes, base_registry
from ..util.pyutil import memoize, deprecated
from ..util._expr import Expr, UnaryWrapper, Symbol
class EyringHS(Expr):
    argument_names = ('dH', 'dS', 'c0')
    argument_defaults = (1 * _molar,)
    parameter_keys = ('temperature', 'molar_gas_constant', 'Boltzmann_constant', 'Planck_constant')

    def args_dimensionality(self, **kwargs):
        return (energy + {'amount': -1}, energy + {'amount': -1, 'temperature': -1}, concentration)

    def __call__(self, variables, backend=math, reaction=None, **kwargs):
        dH, dS, c0 = self.all_args(variables, backend=backend, **kwargs)
        T, R, kB, h = [variables[k] for k in self.parameter_keys]
        return kB / h * T * backend.exp(-(dH - T * dS) / (R * T)) * c0 ** (1 - reaction.order())