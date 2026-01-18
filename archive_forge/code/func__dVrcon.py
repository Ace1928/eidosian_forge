import json
from os.path import join, abspath, dirname
from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
def _dVrcon(m, t):
    if t == 0:
        return Constraint.Skip
    return m.dVr[t] == m.Fa[t] * m.Mwa / m.rhor