from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.simulator import Simulator
from pyomo.contrib.sensitivity_toolbox.sens import sensitivity_calculation
def CostFunc(m):
    return m.L[m.tf]