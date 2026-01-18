from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.simulator import Simulator
from pyomo.contrib.sensitivity_toolbox.sens import sensitivity_calculation
def _yy00(m, t):
    return sum((m.pp[kk] * m.yy[t, kk] for kk in m.ij)) * m.dyy[t, (0, 0)] == sum((m.pp[kk] * m.yy[t, kk] for kk in m.ij)) * (m.II[0, 0] - (m.vp[t] + m.dd[0, 0]) * m.yy[t, (0, 0)] + m.omeg * m.yy[t, (0, 1)]) - m.pp[0, 0] * sum((m.bb[kk] * m.eta00[kk] * m.pp[kk] * m.yy[t, kk] for kk in m.ij)) * m.yy[t, (0, 0)]