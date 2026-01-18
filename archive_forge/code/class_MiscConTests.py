import sys
import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr import (
from pyomo.core.base.constraint import _GeneralConstraintData
class MiscConTests(unittest.TestCase):

    def test_infeasible(self):
        m = ConcreteModel()
        with self.assertRaisesRegex(ValueError, "Constraint 'c' is always infeasible"):
            m.c = Constraint(expr=Constraint.Infeasible)
        self.assertEqual(m.c._data, {})
        with self.assertRaisesRegex(ValueError, "Constraint 'c' is always infeasible"):
            m.c = Constraint.Infeasible
        self.assertEqual(m.c._data, {})
        self.assertIsNone(m.c.expr)
        m.c = (0, 1, 2)
        self.assertIn(None, m.c._data)
        self.assertEqual(m.c.lb, 0)
        self.assertEqual(m.c.ub, 2)
        with self.assertRaisesRegex(ValueError, "Constraint 'c' is always infeasible"):
            m.c = Constraint.Infeasible
        self.assertEqual(m.c._data, {})
        self.assertIsNone(m.c.expr)
        self.assertEqual(m.c.lb, None)
        self.assertEqual(m.c.ub, None)

    def test_slack_methods(self):
        model = ConcreteModel()
        model.x = Var(initialize=2.0)
        L = -1.0
        U = 5.0
        model.cL = Constraint(expr=model.x ** 2 >= L)
        self.assertEqual(model.cL.lslack(), 5.0)
        self.assertEqual(model.cL.uslack(), float('inf'))
        self.assertEqual(model.cL.slack(), 5.0)
        model.cU = Constraint(expr=model.x ** 2 <= U)
        self.assertEqual(model.cU.lslack(), float('inf'))
        self.assertEqual(model.cU.uslack(), 1.0)
        self.assertEqual(model.cU.slack(), 1.0)
        model.cR = Constraint(expr=(L, model.x ** 2, U))
        self.assertEqual(model.cR.lslack(), 5.0)
        self.assertEqual(model.cR.uslack(), 1.0)
        self.assertEqual(model.cR.slack(), 1.0)

    def test_constructor(self):
        a = Constraint(name='b')
        self.assertEqual(a.local_name, 'b')
        try:
            a = Constraint(foo='bar')
            self.fail("Can't specify an unexpected constructor option")
        except ValueError:
            pass

    def test_contains(self):
        model = ConcreteModel()
        model.a = Set(initialize=[1, 2, 3])
        model.b = Constraint(model.a)
        self.assertEqual(2 in model.b, False)
        tmp = []
        for i in model.b:
            tmp.append(i)
        self.assertEqual(len(tmp), 0)

    def test_empty_singleton(self):
        a = Constraint()
        a.construct()
        self.assertEqual(a._constructed, True)
        self.assertEqual(len(a), 0)
        try:
            a()
            self.fail('Component is empty')
        except ValueError:
            pass
        try:
            a.body
            self.fail('Component is empty')
        except ValueError:
            pass
        try:
            a.lower
            self.fail('Component is empty')
        except ValueError:
            pass
        try:
            a.upper
            self.fail('Component is empty')
        except ValueError:
            pass
        try:
            a.equality
            self.fail('Component is empty')
        except ValueError:
            pass
        try:
            a.strict_lower
            self.fail('Component is empty')
        except ValueError:
            pass
        try:
            a.strict_upper
            self.fail('Component is empty')
        except ValueError:
            pass
        x = Var(initialize=1.0)
        x.construct()
        a.set_value((0, x, 2))
        self.assertEqual(len(a), 1)
        self.assertEqual(a(), 1)
        self.assertEqual(a.body(), 1)
        self.assertEqual(a.lower(), 0)
        self.assertEqual(a.upper(), 2)
        self.assertEqual(a.equality, False)
        self.assertEqual(a.strict_lower, False)
        self.assertEqual(a.strict_upper, False)

    def test_unconstructed_singleton(self):
        a = Constraint()
        self.assertEqual(a._constructed, False)
        self.assertEqual(len(a), 0)
        with self.assertRaisesRegex(RuntimeError, 'Cannot access .* on AbstractScalarConstraint.*before it has been constructed'):
            a()
        with self.assertRaisesRegex(RuntimeError, 'Cannot access .* on AbstractScalarConstraint.*before it has been constructed'):
            a.body
        with self.assertRaisesRegex(RuntimeError, 'Cannot access .* on AbstractScalarConstraint.*before it has been constructed'):
            a.lower
        with self.assertRaisesRegex(RuntimeError, 'Cannot access .* on AbstractScalarConstraint.*before it has been constructed'):
            a.upper
        with self.assertRaisesRegex(RuntimeError, 'Cannot access .* on AbstractScalarConstraint.*before it has been constructed'):
            a.equality
        with self.assertRaisesRegex(RuntimeError, 'Cannot access .* on AbstractScalarConstraint.*before it has been constructed'):
            a.strict_lower
        with self.assertRaisesRegex(RuntimeError, 'Cannot access .* on AbstractScalarConstraint.*before it has been constructed'):
            a.strict_upper
        x = Var(initialize=1.0)
        x.construct()
        a.construct()
        a.set_value((0, x, 2))
        self.assertEqual(len(a), 1)
        self.assertEqual(a(), 1)
        self.assertEqual(a.body(), 1)
        self.assertEqual(a.lower(), 0)
        self.assertEqual(a.upper(), 2)
        self.assertEqual(a.equality, False)
        self.assertEqual(a.strict_lower, False)
        self.assertEqual(a.strict_upper, False)

    def test_rule(self):

        def rule1(model):
            return Constraint.Skip
        model = ConcreteModel()
        try:
            model.o = Constraint(rule=rule1)
        except Exception:
            e = sys.exc_info()[1]
            self.fail('Failure to create empty constraint: %s' % str(e))

        def rule1(model):
            return (0.0, model.x, 2.0)
        model = ConcreteModel()
        model.x = Var(initialize=1.1)
        model.o = Constraint(rule=rule1)
        self.assertEqual(model.o(), 1.1)

        def rule1(model, i):
            return Constraint.Skip
        model = ConcreteModel()
        model.a = Set(initialize=[1, 2, 3])
        try:
            model.o = Constraint(model.a, rule=rule1)
        except Exception:
            self.fail('Error generating empty constraint')

        def rule1(model):
            return (0.0, 1.1, 2.0, None)
        model = ConcreteModel()
        try:
            model.o = Constraint(rule=rule1)
            self.fail('Can only return tuples of length 2 or 3')
        except ValueError:
            pass

    def test_tuple_constraint_create(self):

        def rule1(model):
            return (0.0, model.x)
        model = ConcreteModel()
        model.x = Var()
        model.y = Var()
        model.z = Var()
        model.c = Constraint(rule=rule1)
        self.assertEqual(model.c.lower, 0)
        self.assertIs(model.c.body, model.x)
        self.assertEqual(model.c.upper, 0)

        def rule1(model):
            return (model.y, model.x, model.z)
        model = AbstractModel()
        model.x = Var()
        model.y = Var()
        model.z = Var()
        model.c = Constraint(rule=rule1)
        instance = model.create_instance()
        with self.assertRaisesRegex(ValueError, "Constraint 'c' is a Ranged Inequality with a variable lower bound"):
            instance.c.lower
        self.assertIs(instance.c.body, instance.x)
        with self.assertRaisesRegex(ValueError, "Constraint 'c' is a Ranged Inequality with a variable upper bound"):
            instance.c.upper

    def test_expression_constructor_coverage(self):

        def rule1(model):
            expr = model.x
            expr = expr == 0.0
            expr = expr >= 1.0
            return expr
        model = AbstractModel()
        model.x = Var()
        model.y = Var()
        model.z = Var()
        model.o = Constraint(rule=rule1)
        self.assertRaises(TypeError, model.create_instance)

        def rule1(model):
            expr = model.U >= model.x
            expr = expr >= model.L
            return expr
        model = ConcreteModel()
        model.x = Var()
        model.L = Param(initialize=0)
        model.U = Param(initialize=1)
        model.o = Constraint(rule=rule1)

        def rule1(model):
            expr = model.x <= model.z
            expr = expr >= model.y
            return expr
        model = AbstractModel()
        model.x = Var()
        model.y = Var()
        model.z = Var()
        model.o = Constraint(rule=rule1)

        def rule1(model):
            expr = model.x >= model.z
            expr = model.y >= expr
            return expr
        model = AbstractModel()
        model.x = Var()
        model.y = Var()
        model.z = Var()
        model.o = Constraint(rule=rule1)

        def rule1(model):
            expr = model.y <= model.x
            expr = model.y >= expr
            return expr
        model = AbstractModel()
        model.x = Var()
        model.y = Var()
        model.o = Constraint(rule=rule1)

        def rule1(model):
            expr = model.x >= model.L
            return expr
        model = ConcreteModel()
        model.x = Var()
        model.L = Param(initialize=0)
        model.o = Constraint(rule=rule1)

        def rule1(model):
            expr = model.U >= model.x
            return expr
        model = ConcreteModel()
        model.x = Var()
        model.U = Param(initialize=0)
        model.o = Constraint(rule=rule1)

        def rule1(model):
            expr = model.x
            expr = expr == 0.0
            expr = expr <= 1.0
            return expr
        model = AbstractModel()
        model.x = Var()
        model.y = Var()
        model.z = Var()
        model.o = Constraint(rule=rule1)
        self.assertRaises(TypeError, model.create_instance)

        def rule1(model):
            expr = model.U <= model.x
            expr = expr <= model.L
            return expr
        model = ConcreteModel()
        model.x = Var()
        model.L = Param(initialize=0)
        model.U = Param(initialize=1)
        model.o = Constraint(rule=rule1)

        def rule1(model):
            expr = model.x >= model.z
            expr = expr <= model.y
            return expr
        model = AbstractModel()
        model.x = Var()
        model.y = Var()
        model.z = Var()
        model.o = Constraint(rule=rule1)

        def rule1(model):
            expr = model.x <= model.z
            expr = model.y <= expr
            return expr
        model = AbstractModel()
        model.x = Var()
        model.y = Var()
        model.z = Var()
        model.o = Constraint(rule=rule1)

        def rule1(model):
            expr = model.x <= model.L
            return expr
        model = ConcreteModel()
        model.x = Var()
        model.L = Param(initialize=0)
        model.o = Constraint(rule=rule1)

        def rule1(model):
            expr = model.y >= model.x
            expr = model.y <= expr
            return expr
        model = AbstractModel()
        model.x = Var()
        model.y = Var()
        model.o = Constraint(rule=rule1)

        def rule1(model):
            expr = model.U <= model.x
            return expr
        model = ConcreteModel()
        model.x = Var()
        model.U = Param(initialize=0)
        model.o = Constraint(rule=rule1)

        def rule1(model):
            return model.x + model.x
        model = ConcreteModel()
        model.x = Var()
        try:
            model.o = Constraint(rule=rule1)
            self.fail('Cannot return an unbounded expression')
        except ValueError:
            pass

    def test_abstract_index(self):
        model = AbstractModel()
        model.A = Set()
        model.B = Set()
        model.C = model.A | model.B
        model.x = Constraint(model.C)

    def test_ranged_inequality_expr(self):
        model = ConcreteModel()
        model.v = Var()
        model.l = Param(initialize=1, mutable=True)
        model.u = Param(initialize=3, mutable=True)
        model.con = Constraint(expr=inequality(model.l, model.v, model.u))
        self.assertIs(model.con.expr.args[0], model.l)
        self.assertIs(model.con.expr.args[1], model.v)
        self.assertIs(model.con.expr.args[2], model.u)

    def test_potentially_variable_bounds(self):
        m = ConcreteModel()
        m.x = Var()
        m.l = Expression()
        m.u = Expression()
        m.c = Constraint(expr=inequality(m.l, m.x, m.u))
        self.assertIs(m.c.lower, m.l)
        self.assertIs(m.c.upper, m.u)
        with self.assertRaisesRegex(ValueError, 'No value for uninitialized NumericValue object l'):
            m.c.lb
        with self.assertRaisesRegex(ValueError, 'No value for uninitialized NumericValue object u'):
            m.c.ub
        m.l = 5
        m.u = 10
        self.assertIs(m.c.lower, m.l)
        self.assertIs(m.c.upper, m.u)
        self.assertEqual(m.c.lb, 5)
        self.assertEqual(m.c.ub, 10)
        m.l.expr = m.x
        with self.assertRaisesRegex(ValueError, "Constraint 'c' is a Ranged Inequality with a variable lower bound"):
            m.c.lower
        self.assertIs(m.c.upper, m.u)
        with self.assertRaisesRegex(ValueError, "Constraint 'c' is a Ranged Inequality with a variable lower bound"):
            m.c.lb
        self.assertEqual(m.c.ub, 10)
        m.l = 15
        m.u.expr = m.x
        self.assertIs(m.c.lower, m.l)
        with self.assertRaisesRegex(ValueError, "Constraint 'c' is a Ranged Inequality with a variable upper bound"):
            m.c.upper
        self.assertEqual(m.c.lb, 15)
        with self.assertRaisesRegex(ValueError, "Constraint 'c' is a Ranged Inequality with a variable upper bound"):
            m.c.ub
        m.l = -float('inf')
        m.u = float('inf')
        self.assertIs(m.c.lower, m.l)
        self.assertIs(m.c.upper, m.u)
        self.assertIsNone(m.c.lb)
        self.assertIsNone(m.c.ub)
        m.l = float('inf')
        m.u = -float('inf')
        self.assertIs(m.c.lower, m.l)
        self.assertIs(m.c.upper, m.u)
        with self.assertRaisesRegex(ValueError, "Constraint 'c' created with an invalid non-finite lower bound \\(inf\\)"):
            m.c.lb
        with self.assertRaisesRegex(ValueError, "Constraint 'c' created with an invalid non-finite upper bound \\(-inf\\)"):
            m.c.ub
        m.l = float('nan')
        m.u = -float('nan')
        self.assertIs(m.c.lower, m.l)
        self.assertIs(m.c.upper, m.u)
        with self.assertRaisesRegex(ValueError, "Constraint 'c' created with an invalid non-finite lower bound \\(nan\\)"):
            m.c.lb
        with self.assertRaisesRegex(ValueError, "Constraint 'c' created with an invalid non-finite upper bound \\(nan\\)"):
            m.c.ub

    def test_tuple_expression(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.p = Param(mutable=True, initialize=0)
        m.c = Constraint()
        m.c = (m.x, m.y)
        self.assertTrue(m.c.equality)
        self.assertIs(type(m.c.expr), EqualityExpression)
        with self.assertRaisesRegex(ValueError, "Constraint 'c' does not have a proper value. Equality Constraints expressed as 2-tuples cannot contain None"):
            m.c = (m.x, None)
        with self.assertRaisesRegex(ValueError, "Constraint 'c' created with an invalid non-finite lower bound \\(inf\\)"):
            m.c = (m.x, float('inf'))
        with self.assertRaisesRegex(ValueError, "Equality constraint 'c' defined with non-finite term"):
            m.c = EqualityExpression((m.x, None))