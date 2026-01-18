import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from pyomo.common.dependencies.matplotlib import pyplot as plt
def _rate_eqn_rule(m, t, j):
    return m.rate_gen[t, j] - m.stoich[j] * m.k_rxn * m.conc[t, 'A'] == 0