import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from pyomo.common.dependencies.matplotlib import pyplot as plt
def _conc_steady_eqn_rule(m, t, j):
    return m.flow_in[t] * m.conc_in[t, j] - m.flow_out[t] * m.conc_out[t, j] + m.rate_gen[t, j] == 0