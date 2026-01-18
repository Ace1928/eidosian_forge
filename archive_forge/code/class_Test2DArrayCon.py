import sys
import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr import (
from pyomo.core.base.constraint import _GeneralConstraintData
class Test2DArrayCon(unittest.TestCase):

    def create_model(self):
        model = ConcreteModel()
        model.A = Set(initialize=[1, 2])
        return model

    def test_rule_option(self):
        """Test rule option"""
        model = self.create_model()
        model.B = RangeSet(1, 4)

        def f(model, i, j):
            ans = 0
            for j in model.B:
                ans = ans + model.x[j]
            ans *= i
            ans = ans <= 0
            ans = ans >= 0
            return ans
        model.x = Var(model.B, initialize=2)
        model.c = Constraint(model.A, model.A, rule=f)
        self.assertEqual(model.c[1, 1](), 8)
        self.assertEqual(model.c[2, 1](), 16)

    def test_dim(self):
        """Test dim method"""
        model = self.create_model()
        model.c = Constraint(model.A, model.A)
        self.assertEqual(model.c.dim(), 2)

    def test_keys(self):
        """Test keys method"""
        model = self.create_model()
        model.c = Constraint(model.A, model.A)
        self.assertEqual(len(list(model.c.keys())), 0)

    def test_len(self):
        """Test len method"""
        model = self.create_model()
        model.c = Constraint(model.A, model.A)
        self.assertEqual(len(model.c), 0)
        model = self.create_model()
        model.B = RangeSet(1, 4)
        'Test rule option'

        def f(model):
            ans = 0
            for i in model.B:
                ans = ans + model.x[i]
            ans = ans == 2
            return ans
        model.x = Var(model.B, initialize=2)
        model.c = Constraint(rule=f)
        self.assertEqual(len(model.c), 1)