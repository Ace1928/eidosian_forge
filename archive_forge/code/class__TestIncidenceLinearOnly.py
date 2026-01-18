import pyomo.environ as pyo
from pyomo.repn import generate_standard_repn
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.contrib.incidence_analysis.incidence import (
class _TestIncidenceLinearOnly(object):
    """Tests for methods that support linear_only"""

    def _get_incident_variables(self, expr):
        raise NotImplementedError('_TestIncidenceLinearOnly should not be used directly')

    def test_linear_only(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        expr = 2 * m.x[1] + 4 * m.x[2] * m.x[1] - m.x[1] * pyo.exp(m.x[3])
        variables = self._get_incident_variables(expr, linear_only=True)
        self.assertEqual(len(variables), 0)
        expr = 2 * m.x[1] + 2 * m.x[2] * m.x[3] + 3 * m.x[2]
        variables = self._get_incident_variables(expr, linear_only=True)
        self.assertEqual(ComponentSet(variables), ComponentSet([m.x[1]]))
        m.x[3].fix(2.5)
        expr = 2 * m.x[1] + 2 * m.x[2] * m.x[3] + 3 * m.x[2]
        variables = self._get_incident_variables(expr, linear_only=True)
        self.assertEqual(ComponentSet(variables), ComponentSet([m.x[1], m.x[2]]))