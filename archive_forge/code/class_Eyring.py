from collections import OrderedDict
from functools import reduce
import math
from operator import add
from ..units import get_derived_unit, default_units, energy, concentration
from ..util._dimensionality import dimension_codes, base_registry
from ..util.pyutil import memoize, deprecated
from ..util._expr import Expr, UnaryWrapper, Symbol
class Eyring(Expr):
    """Rate expression for Eyring eq: c0*T*exp(-c1/T)

    Note that choice of standard state (c^0) will matter for order > 1.
    """
    argument_names = ('kB_h_times_exp_dS_R', 'dH_over_R', 'conc0')
    argument_defaults = (1 * _molar,)
    parameter_keys = ('temperature',)

    def args_dimensionality(self, reaction):
        order = reaction.order()
        return ({'time': -1, 'temperature': -1, 'amount': 1 - order, 'length': 3 * (order - 1)}, {'temperature': 1}, concentration)

    def __call__(self, variables, backend=math, **kwargs):
        c0, c1, conc0 = self.all_args(variables, backend=backend, **kwargs)
        T = variables['temperature']
        return c0 * T * backend.exp(-c1 / T) * conc0 ** (1 - kwargs['reaction'].order())