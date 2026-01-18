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
class TestReplacementWithNPV(unittest.TestCase):

    def test_npv_sum(self):
        m = ConcreteModel()
        m.p1 = Param(mutable=True)
        m.p2 = Param(mutable=True)
        m.x = Var()
        e1 = m.p1 + 2
        e2 = replace_expressions(e1, {id(m.p1): m.p2})
        e3 = replace_expressions(e1, {id(m.p1): m.x})
        assertExpressionsEqual(self, e2, m.p2 + 2)
        assertExpressionsEqual(self, e3, LinearExpression([MonomialTermExpression((1, m.x)), 2]))

    def test_npv_negation(self):
        m = ConcreteModel()
        m.p1 = Param(mutable=True)
        m.p2 = Param(mutable=True)
        m.x = Var()
        e1 = -m.p1
        e2 = replace_expressions(e1, {id(m.p1): m.p2})
        e3 = replace_expressions(e1, {id(m.p1): m.x})
        assertExpressionsEqual(self, e2, -m.p2)
        assertExpressionsEqual(self, e3, NegationExpression([m.x]))

    def test_npv_pow(self):
        m = ConcreteModel()
        m.p1 = Param(mutable=True)
        m.p2 = Param(mutable=True)
        m.x = Var()
        e1 = m.p1 ** 3
        e2 = replace_expressions(e1, {id(m.p1): m.p2})
        e3 = replace_expressions(e1, {id(m.p1): m.x})
        assertExpressionsEqual(self, e2, m.p2 ** 3)
        assertExpressionsEqual(self, e3, m.x ** 3)

    def test_npv_product(self):
        m = ConcreteModel()
        m.p1 = Param(mutable=True)
        m.p2 = Param(mutable=True)
        m.x = Var()
        e1 = m.p1 * 3
        e2 = replace_expressions(e1, {id(m.p1): m.p2})
        e3 = replace_expressions(e1, {id(m.p1): m.x})
        assertExpressionsEqual(self, e2, m.p2 * 3)
        assertExpressionsEqual(self, e3, ProductExpression([m.x, 3]))

    def test_npv_div(self):
        m = ConcreteModel()
        m.p1 = Param(mutable=True)
        m.p2 = Param(mutable=True)
        m.x = Var()
        e1 = m.p1 / 3
        e2 = replace_expressions(e1, {id(m.p1): m.p2})
        e3 = replace_expressions(e1, {id(m.p1): m.x})
        assertExpressionsEqual(self, e2, m.p2 / 3)
        assertExpressionsEqual(self, e3, DivisionExpression((m.x, 3)))

    def test_npv_unary(self):
        m = ConcreteModel()
        m.p1 = Param(mutable=True)
        m.p2 = Param(mutable=True)
        m.x = Var(initialize=0)
        e1 = sin(m.p1)
        e2 = replace_expressions(e1, {id(m.p1): m.p2})
        e3 = replace_expressions(e1, {id(m.p1): m.x})
        assertExpressionsEqual(self, e2, sin(m.p2))
        assertExpressionsEqual(self, e3, sin(m.x))

    def test_npv_abs(self):
        m = ConcreteModel()
        m.p1 = Param(mutable=True)
        m.p2 = Param(mutable=True)
        m.x = Var()
        e1 = abs(m.p1)
        e2 = replace_expressions(e1, {id(m.p1): m.p2})
        e3 = replace_expressions(e1, {id(m.p1): m.x})
        assertExpressionsEqual(self, e2, abs(m.p2))
        assertExpressionsEqual(self, e3, abs(m.x))