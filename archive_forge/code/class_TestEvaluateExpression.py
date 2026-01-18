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
class TestEvaluateExpression(unittest.TestCase):

    def test_constant(self):
        m = ConcreteModel()
        m.p = Param(initialize=1)
        e = 1 + m.p
        self.assertEqual(2, evaluate_expression(e))
        self.assertEqual(2, evaluate_expression(e, constant=True))

    def test_uninitialized_constant(self):
        m = ConcreteModel()
        m.p = Param(mutable=True)
        e = 1 + m.p
        self.assertRaises(ValueError, evaluate_expression, e)
        self.assertRaises(FixedExpressionError, evaluate_expression, e, constant=True)

    def test_variable(self):
        m = ConcreteModel()
        m.p = Var()
        e = 1 + m.p
        self.assertRaises(ValueError, evaluate_expression, e)
        self.assertRaises(NonConstantExpressionError, evaluate_expression, e, constant=True)

    def test_initialized_variable(self):
        m = ConcreteModel()
        m.p = Var(initialize=1)
        e = 1 + m.p
        self.assertTrue(2, evaluate_expression(e))
        self.assertRaises(NonConstantExpressionError, evaluate_expression, e, constant=True)

    def test_fixed_variable(self):
        m = ConcreteModel()
        m.p = Var(initialize=1)
        m.p.fixed = True
        e = 1 + m.p
        self.assertTrue(2, evaluate_expression(e))
        self.assertRaises(FixedExpressionError, evaluate_expression, e, constant=True)

    def test_template_expr(self):
        m = ConcreteModel()
        m.I = RangeSet(1, 9)
        m.x = Var(m.I, initialize=lambda m, i: i + 1)
        m.P = Param(m.I, initialize=lambda m, i: 10 - i, mutable=True)
        t = IndexTemplate(m.I)
        e = m.x[t + m.P[t + 1]] + 3
        self.assertRaises(TemplateExpressionError, evaluate_expression, e)
        self.assertRaises(TemplateExpressionError, evaluate_expression, e, constant=True)