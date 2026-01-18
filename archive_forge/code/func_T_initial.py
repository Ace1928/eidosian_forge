import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.contrib.doe import ModelOptionLib
def T_initial(m, t):
    if t in m.t_con:
        return controls[t]
    else:
        neighbour_t = max((tc for tc in control_time if tc < t))
        return controls[neighbour_t]