from collections import OrderedDict
from functools import reduce
import math
from operator import add
from ..units import get_derived_unit, default_units, energy, concentration
from ..util._dimensionality import dimension_codes, base_registry
from ..util.pyutil import memoize, deprecated
from ..util._expr import Expr, UnaryWrapper, Symbol
class Arrhenius(Expr):
    """Rate expression for a Arrhenius-type of rate: c0*exp(-c1/T)

    Examples
    --------
    >>> from math import exp
    >>> from chempy import Reaction
    >>> from chempy.units import allclose, default_units as u
    >>> A = 1e11 / u.second
    >>> Ea_over_R = 42e3/8.3145 * u.K**-1
    >>> ratex = MassAction(Arrhenius([A, Ea_over_R]))
    >>> rxn = Reaction({'R'}, {'P'}, ratex)
    >>> dRdt = rxn.rate({'R': 3*u.M, 'temperature': 298.15*u.K})['R']
    >>> allclose(dRdt, -3*1e11*exp(-42e3/8.3145/298.15)*u.M/u.s)
    True

    """
    argument_names = ('A', 'Ea_over_R')
    parameter_keys = ('temperature',)

    def args_dimensionality(self, reaction):
        order = reaction.order()
        return ({'time': -1, 'amount': 1 - order, 'length': 3 * (order - 1)}, {'temperature': 1})

    def __call__(self, variables, backend=math, **kwargs):
        A, Ea_over_R = self.all_args(variables, backend=backend, **kwargs)
        try:
            Ea_over_R = Ea_over_R.simplified
        except AttributeError:
            pass
        return A * backend.exp(-Ea_over_R / variables['temperature'])