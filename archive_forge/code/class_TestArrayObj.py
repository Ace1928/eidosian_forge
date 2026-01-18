import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
class TestArrayObj(unittest.TestCase):

    def create_model(self):
        model = ConcreteModel()
        model.A = Set(initialize=[1, 2])
        return model

    def test_objdata_get_set(self):
        model = ConcreteModel()
        model.o = Objective([1], rule=lambda m, i: 1)
        self.assertEqual(len(model.o), 1)
        self.assertEqual(model.o[1].expr, 1)
        model.o[1].expr = 2
        self.assertEqual(model.o[1].expr, 2)
        model.o[1].expr += 2
        self.assertEqual(model.o[1].expr, 4)

    def test_objdata_get_set_value(self):
        model = ConcreteModel()
        model.o = Objective([1], rule=lambda m, i: 1)
        self.assertEqual(len(model.o), 1)
        self.assertEqual(model.o[1].expr, 1)
        model.o[1].expr = 2
        self.assertEqual(model.o[1].expr, 2)
        model.o[1].expr += 2
        self.assertEqual(model.o[1].expr, 4)

    def test_objdata_get_set_sense(self):
        model = ConcreteModel()
        model.o = Objective([1], rule=lambda m, i: 1, sense=maximize)
        self.assertEqual(len(model.o), 1)
        self.assertEqual(model.o[1].expr, 1)
        self.assertEqual(model.o[1].sense, maximize)
        model.o[1].set_sense(minimize)
        self.assertEqual(model.o[1].sense, minimize)
        model.o[1].sense = maximize
        self.assertEqual(model.o[1].sense, maximize)

    def test_rule_option1(self):
        """Test rule option"""
        model = self.create_model()
        model.B = RangeSet(1, 4)

        def f(model, i):
            ans = 0
            for j in model.B:
                ans = ans + model.x[j]
            ans *= i
            return ans
        model.x = Var(model.B, initialize=2)
        model.obj = Objective(model.A, rule=f)
        self.assertEqual(model.obj[1](), 8)
        self.assertEqual(model.obj[2](), 16)
        self.assertEqual(value(model.obj[1]), 8)
        self.assertEqual(value(model.obj[2]), 16)

    def test_rule_option2(self):
        """Test rule option"""
        model = self.create_model()
        model.B = RangeSet(1, 4)

        def f(model, i):
            if i == 1:
                return Objective.Skip
            ans = 0
            for j in model.B:
                ans = ans + model.x[j]
            ans *= i
            return ans
        model.x = Var(model.B, initialize=2)
        model.obj = Objective(model.A, rule=f)
        self.assertEqual(model.obj[2](), 16)
        self.assertEqual(value(model.obj[2]), 16)

    def test_rule_option3(self):
        """Test rule option"""
        model = self.create_model()
        model.B = RangeSet(1, 4)

        @simple_objective_rule
        def f(model, i):
            if i == 1:
                return None
            ans = 0
            for j in model.B:
                ans = ans + model.x[j]
            ans *= i
            return ans
        model.x = Var(model.B, initialize=2)
        model.obj = Objective(model.A, rule=f)
        self.assertEqual(model.obj[2](), 16)
        self.assertEqual(value(model.obj[2]), 16)

    def test_rule_numeric_expr(self):
        """Test rule option with returns a single numeric constant for the expression"""
        model = self.create_model()

        def f(model, i):
            return 1.0
        model.obj = Objective(model.A, rule=f)
        self.assertEqual(model.obj[2](), 1.0)
        self.assertEqual(value(model.obj[2]), 1.0)

    def test_rule_immutable_param_expr(self):
        """Test rule option that returns a single immutable param for the expression"""
        model = self.create_model()

        def f(model, i):
            return model.p[i]
        model.p = Param(RangeSet(1, 4), initialize=1.0, mutable=False)
        model.x = Var()
        model.obj = Objective(model.A, rule=f)
        self.assertEqual(model.obj[2](), 1.0)
        self.assertEqual(value(model.obj[2]), 1.0)

    def test_rule_mutable_param_expr(self):
        """Test rule option that returns a single mutable param for the expression"""
        model = self.create_model()

        def f(model, i):
            return model.p[i]
        model.r = RangeSet(1, 4)
        model.p = Param(model.r, initialize=1.0, mutable=True)
        model.x = Var()
        model.obj = Objective(model.A, rule=f)
        self.assertEqual(model.obj[2](), 1.0)
        self.assertEqual(value(model.obj[2]), 1.0)

    def test_rule_var_expr(self):
        """Test rule option that returns a single var for the expression"""
        model = self.create_model()

        def f(model, i):
            return model.x[i]
        model.r = RangeSet(1, 4)
        model.x = Var(model.r, initialize=1.0)
        model.obj = Objective(model.A, rule=f)
        self.assertEqual(model.obj[2](), 1.0)
        self.assertEqual(value(model.obj[2]), 1.0)

    def test_sense_option(self):
        """Test sense option"""
        model = self.create_model()
        model.obj1 = Objective(model.A, rule=lambda m, i: 1.0, sense=maximize)
        model.obj2 = Objective(model.A, rule=lambda m, i: 1.0, sense=minimize)
        model.obj3 = Objective(model.A, rule=lambda m, i: 1.0)
        self.assertTrue(len(model.A) > 0)
        self.assertEqual(len(model.obj1), len(model.A))
        self.assertEqual(len(model.obj2), len(model.A))
        self.assertEqual(len(model.obj3), len(model.A))
        for i in model.A:
            self.assertEqual(model.obj1[i].sense, maximize)
            self.assertEqual(model.obj1[i].is_minimizing(), False)
            self.assertEqual(model.obj2[i].sense, minimize)
            self.assertEqual(model.obj2[i].is_minimizing(), True)
            self.assertEqual(model.obj3[i].sense, minimize)
            self.assertEqual(model.obj3[i].is_minimizing(), True)

    def test_dim(self):
        """Test dim method"""
        model = self.create_model()
        model.obj = Objective(model.A)
        self.assertEqual(model.obj.dim(), 1)

    def test_keys(self):
        """Test keys method"""
        model = self.create_model()

        def A_rule(model, i):
            return model.x
        model.x = Var()
        model.obj = Objective(model.A, rule=A_rule)
        self.assertEqual(len(list(model.obj.keys())), 2)

    def test_len(self):
        """Test len method"""
        model = self.create_model()
        model.obj = Objective(model.A)
        self.assertEqual(len(model.obj), 0)
        model = self.create_model()
        'Test rule option'

        def f(model):
            ans = 0
            for i in model.x.keys():
                ans = ans + model.x[i]
            return ans
        model.x = Var(RangeSet(1, 4), initialize=2)
        model.obj = Objective(rule=f)
        self.assertEqual(len(model.obj), 1)