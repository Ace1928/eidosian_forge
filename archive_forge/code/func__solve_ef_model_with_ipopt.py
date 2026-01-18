import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.subsystems import (
from pyomo.common.gsl import find_GSL
def _solve_ef_model_with_ipopt(self):
    m = self._make_model_with_external_functions()
    ipopt = pyo.SolverFactory('ipopt')
    ipopt.solve(m)
    return m