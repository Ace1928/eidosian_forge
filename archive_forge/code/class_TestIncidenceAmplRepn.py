import pyomo.environ as pyo
from pyomo.repn import generate_standard_repn
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.contrib.incidence_analysis.incidence import (
class TestIncidenceAmplRepn(unittest.TestCase, _TestIncidence, _TestIncidenceLinearOnly, _TestIncidenceLinearCancellation):

    def _get_incident_variables(self, expr, **kwds):
        method = IncidenceMethod.ampl_repn
        return get_incident_variables(expr, method=method, **kwds)