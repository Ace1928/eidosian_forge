import os
import platform
import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.numvalue import native_types, nonpyomo_leaf_types, NumericConstant
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.visitor import (
from pyomo.core.base.param import _ParamData, ScalarParam
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.common.collections import ComponentSet
from pyomo.common.errors import TemplateExpressionError
from pyomo.common.log import LoggingIntercept
from io import StringIO
from pyomo.core.expr.compare import assertExpressionsEqual
class TestExpressionUtilities(unittest.TestCase):

    def test_identify_vars_numeric(self):
        self.assertEqual(list(identify_variables(5)), [])

    def test_identify_vars_params(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.a = Param(initialize=1)
        m.b = Param(m.I, initialize=1, mutable=True)
        self.assertEqual(list(identify_variables(m.a)), [])
        self.assertEqual(list(identify_variables(m.b[1])), [])
        self.assertEqual(list(identify_variables(m.a + m.b[1])), [])
        self.assertEqual(list(identify_variables(m.a ** m.b[1])), [])
        self.assertEqual(list(identify_variables(m.a ** m.b[1] + m.b[2])), [])
        self.assertEqual(list(identify_variables(m.a ** m.b[1] + m.b[2] * m.b[3] * m.b[2])), [])

    def test_identify_duplicate_vars(self):
        m = ConcreteModel()
        m.a = Var(initialize=1)
        self.assertEqual(list(identify_variables(2 * m.a + 2 * m.a)), [m.a])

    def test_identify_vars_expr(self):
        m = ConcreteModel()
        m.a = Var(initialize=1)
        m.b = Var(initialize=2)
        m.e = Expression(expr=3 * m.a)
        m.E = Expression([0, 1], initialize={0: 3 * m.a, 1: 4 * m.b})
        self.assertEqual(list(identify_variables(m.b + m.e)), [m.b, m.a])
        self.assertEqual(list(identify_variables(m.E[0])), [m.a])
        self.assertEqual(list(identify_variables(m.E[1])), [m.b])

    def test_identify_vars_vars(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.a = Var(initialize=1)
        m.b = Var(m.I, initialize=1)
        m.p = Param(initialize=1, mutable=True)
        m.x = ExternalFunction(library='foo.so', function='bar')
        self.assertEqual(list(identify_variables(m.a)), [m.a])
        self.assertEqual(list(identify_variables(m.b[1])), [m.b[1]])
        self.assertEqual(list(identify_variables(m.a + m.b[1])), [m.a, m.b[1]])
        self.assertEqual(list(identify_variables(m.a ** m.b[1])), [m.a, m.b[1]])
        self.assertEqual(list(identify_variables(m.a ** m.b[1] + m.b[2])), [m.b[2], m.a, m.b[1]])
        self.assertEqual(list(identify_variables(m.a ** m.b[1] + m.b[2] * m.b[3] * m.b[2])), [m.a, m.b[1], m.b[2], m.b[3]])
        self.assertEqual(list(identify_variables(m.a ** m.b[1] + m.b[2] / m.b[3] * m.b[2])), [m.a, m.b[1], m.b[2], m.b[3]])
        self.assertEqual(list(identify_variables(m.x(m.a, 'string_param', 1, []) * m.b[1])), [m.b[1], m.a])
        self.assertEqual(list(identify_variables(m.x(m.p, 'string_param', 1, []) * m.b[1])), [m.b[1]])
        self.assertEqual(list(identify_variables(tanh(m.a) * m.b[1])), [m.b[1], m.a])
        self.assertEqual(list(identify_variables(abs(m.a) * m.b[1])), [m.b[1], m.a])
        self.assertEqual(list(identify_variables(m.a ** m.a + m.a)), [m.a])

    def test_identify_vars_linear_expression(self):
        m = ConcreteModel()
        m.x = Var()
        expr = quicksum([m.x, m.x], linear=True)
        self.assertEqual(list(identify_variables(expr, include_fixed=False)), [m.x])