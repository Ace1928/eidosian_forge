from collections import OrderedDict
from functools import reduce
import math
from operator import add
from ..units import get_derived_unit, default_units, energy, concentration
from ..util._dimensionality import dimension_codes, base_registry
from ..util.pyutil import memoize, deprecated
from ..util._expr import Expr, UnaryWrapper, Symbol
class SinTemp(Expr):
    argument_names = ('Tbase', 'Tamp', 'angvel', 'phase')
    parameter_keys = ('time',)

    def args_dimensionality(self, **kwargs):
        return ({'temperature': 1}, {'temperature': 1}, {'time': -1}, {})

    def __call__(self, variables, backend=math, **kwargs):
        Tbase, Tamp, angvel, phase = self.all_args(variables, backend=backend, **kwargs)
        return Tbase + Tamp * backend.sin(angvel * variables['time'] + phase)