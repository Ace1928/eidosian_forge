import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.contrib.cp.repn.docplex_writer import docplex_available, LogicalToDoCplex
from pyomo.core.base.range import NumericRange
from pyomo.core.expr.numeric_expr import MinExpression, MaxExpression
from pyomo.core.expr.logical_expr import (
from pyomo.core.expr.relational_expr import NotEqualExpression
from pyomo.environ import (
@unittest.skipIf(not docplex_available, 'docplex is not available')
class TestCPExpressionWalker_AlgebraicExpressions(CommonTest):

    def test_write_addition(self):
        m = self.get_model()
        m.c = Constraint(expr=m.x + m.i.start_time + m.i2[2].length <= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))
        self.assertIn(id(m.x), visitor.var_map)
        self.assertIn(id(m.i), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        self.assertIn(id(m.i2[2].length), visitor.var_map)
        cpx_x = visitor.var_map[id(m.x)]
        cpx_i = visitor.var_map[id(m.i)]
        cpx_i2 = visitor.var_map[id(m.i2[2])]
        self.assertTrue(expr[1].equals(cpx_x + cp.start_of(cpx_i) + cp.length_of(cpx_i2)))

    def test_write_subtraction(self):
        m = self.get_model()
        m.a.domain = Binary
        m.c = Constraint(expr=m.x - m.a[1] <= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))
        self.assertIn(id(m.x), visitor.var_map)
        self.assertIn(id(m.a[1]), visitor.var_map)
        x = visitor.var_map[id(m.x)]
        a1 = visitor.var_map[id(m.a[1])]
        self.assertTrue(expr[1].equals(x + -1 * a1))

    def test_write_product(self):
        m = self.get_model()
        m.a.domain = PositiveIntegers
        m.c = Constraint(expr=m.x * (m.a[1] + 1) <= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))
        self.assertIn(id(m.x), visitor.var_map)
        self.assertIn(id(m.a[1]), visitor.var_map)
        x = visitor.var_map[id(m.x)]
        a1 = visitor.var_map[id(m.a[1])]
        self.assertTrue(expr[1].equals(x * (a1 + 1)))

    def test_write_floating_point_division(self):
        m = self.get_model()
        m.a.domain = NonNegativeIntegers
        m.c = Constraint(expr=m.x / (m.a[1] + 1) <= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))
        self.assertIn(id(m.x), visitor.var_map)
        self.assertIn(id(m.a[1]), visitor.var_map)
        x = visitor.var_map[id(m.x)]
        a1 = visitor.var_map[id(m.a[1])]
        self.assertTrue(expr[1].equals(x / (a1 + 1)))

    def test_write_power_expression(self):
        m = self.get_model()
        m.c = Constraint(expr=m.x ** 2 <= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))
        self.assertIn(id(m.x), visitor.var_map)
        cpx_x = visitor.var_map[id(m.x)]
        self.assertTrue(expr[1].equals(cpx_x ** 2))

    def test_write_absolute_value_expression(self):
        m = self.get_model()
        m.a.domain = NegativeIntegers
        m.c = Constraint(expr=abs(m.a[1]) + 1 <= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))
        self.assertIn(id(m.a[1]), visitor.var_map)
        a1 = visitor.var_map[id(m.a[1])]
        self.assertTrue(expr[1].equals(cp.abs(a1) + 1))

    def test_write_min_expression(self):
        m = self.get_model()
        m.a.domain = NonPositiveIntegers
        m.c = Constraint(expr=MinExpression([m.a[i] for i in m.I]) >= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))
        a = {}
        for i in m.I:
            self.assertIn(id(m.a[i]), visitor.var_map)
            a[i] = visitor.var_map[id(m.a[i])]
        self.assertTrue(expr[1].equals(cp.min((a[i] for i in m.I))))

    def test_write_max_expression(self):
        m = self.get_model()
        m.a.domain = NonPositiveIntegers
        m.c = Constraint(expr=MaxExpression([m.a[i] for i in m.I]) >= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))
        a = {}
        for i in m.I:
            self.assertIn(id(m.a[i]), visitor.var_map)
            a[i] = visitor.var_map[id(m.a[i])]
        self.assertTrue(expr[1].equals(cp.max((a[i] for i in m.I))))

    def test_expression_with_mutable_param(self):
        m = ConcreteModel()
        m.x = Var(domain=Integers, bounds=(2, 3))
        m.p = Param(initialize=4, mutable=True)
        e = m.p * m.x
        visitor = self.get_visitor()
        expr = visitor.walk_expression((e, e, 0))
        self.assertIn(id(m.x), visitor.var_map)
        x = visitor.var_map[id(m.x)]
        self.assertTrue(expr[1].equals(4 * x))