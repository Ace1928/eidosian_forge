import pyomo.environ as pyo
from pyomo.repn import generate_standard_repn
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.contrib.incidence_analysis.incidence import (
class TestIncidenceIdentifyVariables(unittest.TestCase, _TestIncidence):

    def _get_incident_variables(self, expr, **kwds):
        method = IncidenceMethod.identify_variables
        return get_incident_variables(expr, method=method, **kwds)

    def test_zero_coef(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        expr = 0 * m.x[1] + 0 * m.x[1] * m.x[2] + 0 * pyo.exp(m.x[3])
        variables = self._get_incident_variables(expr)
        self.assertEqual(ComponentSet(variables), ComponentSet(m.x[:]))

    def test_variable_minus_itself(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        expr = m.x[1] + m.x[2] * m.x[3] - m.x[1]
        variables = self._get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(var_set, ComponentSet(m.x[:]))

    def test_incidence_with_mutable_parameter(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        m.p = pyo.Param(mutable=True, initialize=None)
        expr = m.x[1] + m.p * m.x[1] * m.x[2] + m.x[1] * pyo.exp(m.x[3])
        variables = self._get_incident_variables(expr)
        self.assertEqual(ComponentSet(variables), ComponentSet(m.x[:]))