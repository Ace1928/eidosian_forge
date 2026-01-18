from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.simulator import Simulator
from pyomo.contrib.sensitivity_toolbox.sens import sensitivity_calculation
def _yy30(m, t):
    return m.dyy[t, (3, 0)] == m.II[3, 0] + m.dd[1, 0] * m.yy[t, (1, 0)] + m.dd[1, 1] * m.yy[t, (1, 1)] + m.dd[2, 0] * m.yy[t, (2, 0)] + m.dd[2, 1] * m.yy[t, (2, 1)] - (m.dd[3, 0] + m.dd[0, 0]) * m.yy[t, (3, 0)]