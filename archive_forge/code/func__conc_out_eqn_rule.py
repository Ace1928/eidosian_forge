import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from pyomo.common.dependencies.matplotlib import pyplot as plt
def _conc_out_eqn_rule(m, t, j):
    return m.conc[t, j] - m.conc_out[t, j] == 0