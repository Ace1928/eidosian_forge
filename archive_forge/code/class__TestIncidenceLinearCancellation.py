import pyomo.environ as pyo
from pyomo.repn import generate_standard_repn
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.contrib.incidence_analysis.incidence import (
class _TestIncidenceLinearCancellation(object):
    """Tests for methods that perform linear cancellation"""

    def _get_incident_variables(self, expr):
        raise NotImplementedError('_TestIncidenceLinearCancellation should not be used directly')

    def test_zero_coef(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        expr = 0 * m.x[1] + 0 * m.x[1] * m.x[2] + 0 * pyo.exp(m.x[3])
        variables = self._get_incident_variables(expr)
        self.assertEqual(len(variables), 0)

    def test_variable_minus_itself(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        expr = m.x[1] + m.x[2] * m.x[3] - m.x[1]
        variables = self._get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(var_set, ComponentSet([m.x[2], m.x[3]]))

    def test_fixed_zero_linear_coefficient(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        m.p = pyo.Param([1, 2], mutable=True, initialize=1.0)
        m.p[1].set_value(0)
        expr = 2 * m.x[1] + m.p[1] * m.p[2] * m.x[2] + m.p[2] * m.x[3] ** 2
        variables = self._get_incident_variables(expr)
        self.assertEqual(ComponentSet(variables), ComponentSet([m.x[1], m.x[3]]))
        m.x[3].fix(0.0)
        expr = 2 * m.x[1] + 3 * m.x[3] * m.p[2] * m.x[2] + m.x[1] ** 2
        variables = self._get_incident_variables(expr)
        self.assertEqual(ComponentSet(variables), ComponentSet([m.x[1]]))
        m.x[3].fix(1.0)
        variables = self._get_incident_variables(expr)
        self.assertEqual(ComponentSet(variables), ComponentSet([m.x[1], m.x[2]]))

    def test_fixed_zero_coefficient_linear_only(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        expr = m.x[1] * m.x[2] + 2 * m.x[3]
        m.x[2].fix(0)
        variables = get_incident_variables(expr, method=IncidenceMethod.standard_repn, linear_only=True)
        self.assertEqual(len(variables), 1)
        self.assertIs(variables[0], m.x[3])