from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.simulator import Simulator
from pyomo.contrib.sensitivity_toolbox.sens import sensitivity_calculation
def _yy21(m, t):
    return m.dyy[t, (2, 1)] == m.vt[t] * m.yy[t, (2, 0)] - (m.dd[2, 1] + m.dd[0, 0]) * m.yy[t, (2, 1)]