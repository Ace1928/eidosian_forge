import sys
import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr import (
from pyomo.core.base.constraint import _GeneralConstraintData
class TestSimpleCon(unittest.TestCase):

    def test_set_expr_explicit_multivariate(self):
        """Test expr= option (multivariate expression)"""
        model = ConcreteModel()
        model.A = RangeSet(1, 4)
        model.x = Var(model.A, initialize=2)
        ans = 0
        for i in model.A:
            ans = ans + model.x[i]
        ans = ans >= 0
        ans = ans <= 1
        model.c = Constraint(expr=ans)
        self.assertEqual(model.c(), 8)
        self.assertEqual(model.c.body(), 8)
        self.assertEqual(value(model.c.body), 8)

    def test_set_expr_explicit_univariate(self):
        """Test expr= option (univariate expression)"""
        model = ConcreteModel()
        model.x = Var(initialize=2)
        ans = model.x >= 0
        ans = ans <= 1
        model.c = Constraint(expr=ans)
        self.assertEqual(model.c(), 2)
        self.assertEqual(model.c.body(), 2)
        self.assertEqual(value(model.c.body), 2)

    def test_set_expr_undefined_univariate(self):
        """Test expr= option (univariate expression)"""
        model = ConcreteModel()
        model.x = Var()
        ans = model.x >= 0
        ans = ans <= 1
        model.c = Constraint(expr=ans)
        with self.assertRaisesRegex(ValueError, 'No value for uninitialized NumericValue object x'):
            value(model.c)
        model.x = 2
        self.assertEqual(model.c(), 2)
        self.assertEqual(value(model.c.body), 2)

    def test_set_expr_inline(self):
        """Test expr= option (inline expression)"""
        model = ConcreteModel()
        model.A = RangeSet(1, 4)
        model.x = Var(model.A, initialize=2)
        model.c = Constraint(expr=(0, sum((model.x[i] for i in model.A)), 1))
        self.assertEqual(model.c(), 8)
        self.assertEqual(value(model.c.body), 8)

    def test_rule1(self):
        """Test rule option"""
        model = ConcreteModel()
        model.B = RangeSet(1, 4)

        def f(model):
            ans = 0
            for i in model.B:
                ans = ans + model.x[i]
            ans = ans >= 0
            ans = ans <= 1
            return ans
        model.x = Var(model.B, initialize=2)
        model.c = Constraint(rule=f)
        self.assertEqual(model.c(), 8)
        self.assertEqual(value(model.c.body), 8)

    def test_rule2(self):
        """Test rule option"""
        model = ConcreteModel()
        model.B = RangeSet(1, 4)

        def f(model):
            ans = 0
            for i in model.B:
                ans = ans + model.x[i]
            return (0, ans, 1)
        model.x = Var(model.B, initialize=2)
        model.c = Constraint(rule=f)
        self.assertEqual(model.c(), 8)
        self.assertEqual(value(model.c.body), 8)

    def test_rule3(self):
        """Test rule option"""
        model = ConcreteModel()
        model.B = RangeSet(1, 4)

        def f(model):
            ans = 0
            for i in model.B:
                ans = ans + model.x[i]
            return (0, ans, None)
        model.x = Var(model.B, initialize=2)
        model.c = Constraint(rule=f)
        self.assertEqual(model.c(), 8)
        self.assertEqual(value(model.c.body), 8)

    def test_rule4(self):
        """Test rule option"""
        model = ConcreteModel()
        model.B = RangeSet(1, 4)

        def f(model):
            ans = 0
            for i in model.B:
                ans = ans + model.x[i]
            return (None, ans, 1)
        model.x = Var(model.B, initialize=2)
        model.c = Constraint(rule=f)
        self.assertEqual(model.c(), 8)
        self.assertEqual(value(model.c.body), 8)

    def test_rule5(self):
        """Test rule option"""
        model = ConcreteModel()
        model.B = RangeSet(1, 4)

        def f(model):
            ans = 0
            for i in model.B:
                ans = ans + model.x[i]
            return (ans, 1)
        model.x = Var(model.B, initialize=2)
        model.c = Constraint(rule=f)
        self.assertEqual(model.c(), 8)
        self.assertEqual(value(model.c.body), 8)

    def test_dim(self):
        """Test dim method"""
        model = ConcreteModel()
        model.c = Constraint()
        self.assertEqual(model.c.dim(), 0)

    def test_keys_empty(self):
        """Test keys method"""
        model = ConcreteModel()
        model.c = Constraint()
        self.assertEqual(list(model.c.keys()), [])

    def test_len_empty(self):
        """Test len method"""
        model = ConcreteModel()
        model.c = Constraint()
        self.assertEqual(len(model.c), 0)

    def test_None_key(self):
        """Test keys method"""
        model = ConcreteModel()
        model.x = Var()
        model.c = Constraint(expr=model.x == 1)
        self.assertEqual(list(model.c.keys()), [None])
        self.assertEqual(id(model.c), id(model.c[None]))

    def test_len(self):
        """Test len method"""
        model = AbstractModel()
        model.x = Var()
        model.c = Constraint(rule=lambda m: m.x == 1)
        self.assertEqual(len(model.c), 0)
        inst = model.create_instance()
        self.assertEqual(len(inst.c), 1)

    def test_setitem(self):
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint()
        self.assertEqual(len(m.c), 0)
        m.c = m.x ** 2 <= 4
        self.assertEqual(len(m.c), 1)
        self.assertEqual(list(m.c.keys()), [None])
        self.assertEqual(m.c.upper, 4)
        m.c = Constraint.Skip
        self.assertEqual(len(m.c), 0)