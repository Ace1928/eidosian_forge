from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.simulator import Simulator
from pyomo.contrib.sensitivity_toolbox.sens import sensitivity_calculation
from pyomo.common.dependencies.matplotlib import pyplot as plt
def _Cost(m):
    return 0.5 * m.H * m.x[m.T] ** 2 + m.F[m.T]