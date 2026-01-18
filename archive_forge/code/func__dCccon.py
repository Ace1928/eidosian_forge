import json
from os.path import join, abspath, dirname
from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
def _dCccon(m, t):
    if t == 0:
        return Constraint.Skip
    return m.dCc[t] == m.k2 * exp(-m.E2 / (m.R * m.Tr[t])) * m.Cb[t]