from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.simulator import Simulator
from pyomo.contrib.sensitivity_toolbox.sens import sensitivity_calculation
from pyomo.common.dependencies.matplotlib import pyplot as plt
def _f0(m, t):
    return m.df0[t] == 0.25 * m.u[t] ** 2