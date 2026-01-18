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
class TestCPExpressionWalker_Vars(CommonTest):

    def test_complain_about_non_integer_vars(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.i.is_present.implies(m.a[1] == 5))
        visitor = self.get_visitor()
        with self.assertRaisesRegex(ValueError, "The LogicalToDoCplex writer can only support integer- or Boolean-valued variables. Cannot write Var 'a\\[1\\]' with domain 'Reals'"):
            expr = visitor.walk_expression((m.c.expr, m.c, 0))

    def test_fixed_integer_var(self):
        m = self.get_model()
        m.a.domain = Integers
        m.a[1].fix(3)
        m.c = Constraint(expr=m.a[1] + m.a[2] >= 4)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))
        self.assertIn(id(m.a[2]), visitor.var_map)
        a2 = visitor.var_map[id(m.a[2])]
        self.assertTrue(expr[1].equals(3 + a2))

    def test_fixed_boolean_var(self):
        m = self.get_model()
        m.b.fix(False)
        m.b2['a'].fix(True)
        m.c = LogicalConstraint(expr=m.b.lor(m.b2['a'].land(m.b2['b'])))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))
        self.assertIn(id(m.b2['b']), visitor.var_map)
        b2b = visitor.var_map[id(m.b2['b'])]
        self.assertTrue(expr[1].equals(cp.logical_or(False, cp.logical_and(True, b2b))))

    def test_indirection_single_index(self):
        m = self.get_model()
        m.a.domain = Integers
        m.c = Constraint(expr=m.a[m.x] >= 3.5)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))
        self.assertIn(id(m.x), visitor.var_map)
        x = visitor.var_map[id(m.x)]
        a = []
        for idx in [6, 7, 8]:
            v = m.a[idx]
            self.assertIn(id(v), visitor.var_map)
            a.append(visitor.var_map[id(v)])
        self.assertTrue(expr[1].equals(cp.element(a, 0 + 1 * (x - 6) // 1)))

    def test_indirection_multi_index_second_constant(self):
        m = self.get_model()
        m.z = Var(m.I, m.I, domain=Integers)
        e = m.z[m.x, 3]
        visitor = self.get_visitor()
        expr = visitor.walk_expression((e, e, 0))
        z = {}
        for i in [6, 7, 8]:
            self.assertIn(id(m.z[i, 3]), visitor.var_map)
            z[i, 3] = visitor.var_map[id(m.z[i, 3])]
        self.assertIn(id(m.x), visitor.var_map)
        x = visitor.var_map[id(m.x)]
        self.assertTrue(expr[1].equals(cp.element([z[i, 3] for i in [6, 7, 8]], 0 + 1 * (x - 6) // 1)))

    def test_indirection_multi_index_first_constant(self):
        m = self.get_model()
        m.z = Var(m.I, m.I, domain=Integers)
        e = m.z[3, m.x]
        visitor = self.get_visitor()
        expr = visitor.walk_expression((e, e, 0))
        z = {}
        for i in [6, 7, 8]:
            self.assertIn(id(m.z[3, i]), visitor.var_map)
            z[3, i] = visitor.var_map[id(m.z[3, i])]
        self.assertIn(id(m.x), visitor.var_map)
        x = visitor.var_map[id(m.x)]
        self.assertTrue(expr[1].equals(cp.element([z[3, i] for i in [6, 7, 8]], 0 + 1 * (x - 6) // 1)))

    def test_indirection_multi_index_neither_constant_same_var(self):
        m = self.get_model()
        m.z = Var(m.I, m.I, domain=Integers)
        e = m.z[m.x, m.x]
        visitor = self.get_visitor()
        expr = visitor.walk_expression((e, e, 0))
        z = {}
        for i in [6, 7, 8]:
            for j in [6, 7, 8]:
                self.assertIn(id(m.z[i, j]), visitor.var_map)
                z[i, j] = visitor.var_map[id(m.z[i, j])]
        self.assertIn(id(m.x), visitor.var_map)
        x = visitor.var_map[id(m.x)]
        self.assertTrue(expr[1].equals(cp.element([z[i, j] for i in [6, 7, 8] for j in [6, 7, 8]], 0 + 1 * (x - 6) // 1 + 3 * (x - 6) // 1)))

    def test_indirection_multi_index_neither_constant_diff_vars(self):
        m = self.get_model()
        m.z = Var(m.I, m.I, domain=Integers)
        m.y = Var(within=[1, 3, 5])
        e = m.z[m.x, m.y]
        visitor = self.get_visitor()
        expr = visitor.walk_expression((e, e, 0))
        z = {}
        for i in [6, 7, 8]:
            for j in [1, 3, 5]:
                self.assertIn(id(m.z[i, 3]), visitor.var_map)
                z[i, j] = visitor.var_map[id(m.z[i, j])]
        self.assertIn(id(m.x), visitor.var_map)
        x = visitor.var_map[id(m.x)]
        self.assertIn(id(m.y), visitor.var_map)
        y = visitor.var_map[id(m.y)]
        self.assertTrue(expr[1].equals(cp.element([z[i, j] for i in [6, 7, 8] for j in [1, 3, 5]], 0 + 1 * (x - 6) // 1 + 3 * (y - 1) // 2)))

    def test_indirection_expression_index(self):
        m = self.get_model()
        m.a.domain = Integers
        m.y = Var(within=[1, 3, 5])
        e = m.a[m.x - m.y]
        visitor = self.get_visitor()
        expr = visitor.walk_expression((e, e, 0))
        a = {}
        for i in range(1, 8):
            self.assertIn(id(m.a[i]), visitor.var_map)
            a[i] = visitor.var_map[id(m.a[i])]
        self.assertIn(id(m.x), visitor.var_map)
        x = visitor.var_map[id(m.x)]
        self.assertIn(id(m.y), visitor.var_map)
        y = visitor.var_map[id(m.y)]
        self.assertTrue(expr[1].equals(cp.element([a[i] for i in range(1, 8)], 0 + 1 * (x + -1 * y - 1) // 1)))

    def test_indirection_fails_with_non_finite_index_domain(self):
        m = self.get_model()
        m.a.domain = Integers
        m.x.setlb(None)
        m.x.setub(None)
        m.c = Constraint(expr=m.a[m.x] >= 0)
        visitor = self.get_visitor()
        with self.assertRaisesRegex(ValueError, "Variable indirection 'a\\[x\\]' contains argument 'x', which is not restricted to a finite discrete domain"):
            expr = visitor.walk_expression((m.c.body, m.c, 0))

    def test_indirection_invalid_index_domain(self):
        m = self.get_model()
        m.a.domain = Integers
        m.a.bounds = (6, 8)
        m.y = Var(within=Integers, bounds=(0, 10))
        e = m.a[m.y]
        visitor = self.get_visitor()
        with self.assertRaisesRegex(ValueError, "Variable indirection 'a\\[y\\]' permits an index '0' that is not a valid key."):
            expr = visitor.walk_expression((e, e, 0))

    def test_infinite_domain_var(self):
        m = ConcreteModel()
        m.Evens = RangeSet(ranges=(NumericRange(0, None, 2), NumericRange(0, None, -2)))
        m.x = Var(domain=m.Evens)
        e = m.x ** 2
        visitor = self.get_visitor()
        with self.assertRaisesRegex(ValueError, "The LogicalToDoCplex writer does not support infinite discrete domains. Cannot write Var 'x' with domain 'Evens'"):
            expr = visitor.walk_expression((e, e, 0))