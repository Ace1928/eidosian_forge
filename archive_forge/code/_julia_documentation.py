from collections import OrderedDict
from chempy import Reaction
from chempy.kinetics.rates import MassAction, RadiolyticBase
from chempy.units import to_unitless, default_units as u

    Parameters
    ==========
    ...
    variables: dict
        e.g. dict(doserate=99.9*u.Gy/u.s, density=998*u.kg/u.m3)
    