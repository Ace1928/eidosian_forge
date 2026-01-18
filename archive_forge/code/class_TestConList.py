import sys
import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr import (
from pyomo.core.base.constraint import _GeneralConstraintData
class TestConList(unittest.TestCase):

    def create_model(self):
        model = ConcreteModel()
        model.A = Set(initialize=[1, 2, 3, 4])
        return model

    def test_conlist_skip(self):
        model = ConcreteModel()
        model.x = Var()
        model.c = ConstraintList()
        self.assertTrue(1 not in model.c)
        self.assertEqual(len(model.c), 0)
        model.c.add(Constraint.Skip)
        self.assertTrue(1 not in model.c)
        self.assertEqual(len(model.c), 0)
        model.c.add(model.x >= 1)
        self.assertTrue(1 not in model.c)
        self.assertTrue(2 in model.c)
        self.assertEqual(len(model.c), 1)

    def test_rule_option1(self):
        model = self.create_model()
        model.B = RangeSet(1, 4)

        def f(model, i):
            if i > 4:
                return ConstraintList.End
            ans = 0
            for j in model.B:
                ans = ans + model.x[j]
            ans *= i
            ans = ans <= 0
            ans = ans >= 0
            return ans
        model.x = Var(model.B, initialize=2)
        model.c = ConstraintList(rule=f)
        self.assertEqual(model.c[1](), 8)
        self.assertEqual(model.c[2](), 16)
        self.assertEqual(len(model.c), 4)

    def test_rule_option2(self):
        model = self.create_model()
        model.B = RangeSet(1, 4)

        def f(model, i):
            if i > 2:
                return ConstraintList.End
            i = 2 * i - 1
            ans = 0
            for j in model.B:
                ans = ans + model.x[j]
            ans *= i
            ans = ans <= 0
            ans = ans >= 0
            return ans
        model.x = Var(model.B, initialize=2)
        model.c = ConstraintList(rule=f)
        self.assertEqual(model.c[1](), 8)
        self.assertEqual(len(model.c), 2)

    def test_rule_option1a(self):
        model = self.create_model()
        model.B = RangeSet(1, 4)

        @simple_constraintlist_rule
        def f(model, i):
            if i > 4:
                return None
            ans = 0
            for j in model.B:
                ans = ans + model.x[j]
            ans *= i
            ans = ans <= 0
            ans = ans >= 0
            return ans
        model.x = Var(model.B, initialize=2)
        model.c = ConstraintList(rule=f)
        self.assertEqual(model.c[1](), 8)
        self.assertEqual(model.c[2](), 16)
        self.assertEqual(len(model.c), 4)

    def test_rule_option2a(self):
        model = self.create_model()
        model.B = RangeSet(1, 4)

        @simple_constraintlist_rule
        def f(model, i):
            if i > 2:
                return None
            i = 2 * i - 1
            ans = 0
            for j in model.B:
                ans = ans + model.x[j]
            ans *= i
            ans = ans <= 0
            ans = ans >= 0
            return ans
        model.x = Var(model.B, initialize=2)
        model.c = ConstraintList(rule=f)
        self.assertEqual(model.c[1](), 8)
        self.assertEqual(len(model.c), 2)

    def test_rule_option3(self):
        model = self.create_model()
        model.y = Var(initialize=2)

        def f(model):
            yield (model.y <= 0)
            yield (2 * model.y <= 0)
            yield (2 * model.y <= 0)
            yield ConstraintList.End
        model.c = ConstraintList(rule=f)
        self.assertEqual(len(model.c), 3)
        self.assertEqual(model.c[1](), 2)
        model.d = ConstraintList(rule=f(model))
        self.assertEqual(len(model.d), 3)
        self.assertEqual(model.d[1](), 2)

    def test_rule_option4(self):
        model = self.create_model()
        model.y = Var(initialize=2)
        model.c = ConstraintList(rule=((i + 1) * model.y >= 0 for i in range(3)))
        self.assertEqual(len(model.c), 3)
        self.assertEqual(model.c[1](), 2)

    def test_dim(self):
        model = self.create_model()
        model.c = ConstraintList()
        self.assertEqual(model.c.dim(), 1)

    def test_keys(self):
        model = self.create_model()
        model.c = ConstraintList()
        self.assertEqual(len(list(model.c.keys())), 0)

    def test_len(self):
        model = self.create_model()
        model.c = ConstraintList()
        self.assertEqual(len(model.c), 0)

    def test_0based_add(self):
        m = ConcreteModel()
        m.x = Var()
        m.c = ConstraintList(starting_index=0)
        m.c.add(m.x <= 0)
        self.assertEqual(list(m.c.keys()), [0])
        m.c.add(m.x >= 0)
        self.assertEqual(list(m.c.keys()), [0, 1])