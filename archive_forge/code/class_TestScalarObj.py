import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
class TestScalarObj(unittest.TestCase):

    def test_singleton_get_set(self):
        model = ConcreteModel()
        model.o = Objective(expr=1)
        self.assertEqual(len(model.o), 1)
        self.assertEqual(model.o.expr, 1)
        model.o.expr = 2
        self.assertEqual(model.o.expr, 2)
        model.o.expr += 2
        self.assertEqual(model.o.expr, 4)

    def test_singleton_get_set_value(self):
        model = ConcreteModel()
        model.o = Objective(expr=1)
        self.assertEqual(len(model.o), 1)
        self.assertEqual(model.o.expr, 1)
        model.o.expr = 2
        self.assertEqual(model.o.expr, 2)
        model.o.expr += 2
        self.assertEqual(model.o.expr, 4)

    def test_scalar_invalid_expr(self):
        m = ConcreteModel()
        m.x = Var()
        with self.assertRaisesRegex(ValueError, "Cannot assign InequalityExpression to 'obj': ScalarObjective components only allow numeric expression types."):
            m.obj = Objective(expr=m.x <= 0)

    def test_empty_singleton(self):
        a = Objective()
        a.construct()
        self.assertEqual(a._constructed, True)
        self.assertEqual(len(a), 0)
        try:
            a()
            self.fail('Component is empty')
        except ValueError:
            pass
        try:
            a.expr
            self.fail('Component is empty')
        except ValueError:
            pass
        try:
            a.sense
            self.fail('Component is empty')
        except ValueError:
            pass
        x = Var(initialize=1.0)
        x.construct()
        a.set_value(x + 1)
        self.assertEqual(len(a), 1)
        self.assertEqual(a(), 2)
        self.assertEqual(a.expr(), 2)
        self.assertEqual(a.sense, minimize)

    def test_unconstructed_singleton(self):
        a = Objective()
        self.assertEqual(a._constructed, False)
        self.assertEqual(len(a), 0)
        try:
            a()
            self.fail('Component is unconstructed')
        except ValueError:
            pass
        try:
            a.expr
            self.fail('Component is unconstructed')
        except ValueError:
            pass
        try:
            a.sense
            self.fail('Component is unconstructed')
        except ValueError:
            pass
        a.construct()
        a.set_sense(minimize)
        self.assertEqual(len(a), 1)
        self.assertEqual(a(), None)
        self.assertEqual(a.expr, None)
        self.assertEqual(a.sense, minimize)
        a.sense = maximize
        self.assertEqual(len(a), 1)
        self.assertEqual(a(), None)
        self.assertEqual(a.expr, None)
        self.assertEqual(a.sense, maximize)

    def test_numeric_expr(self):
        """Test expr option with a single numeric constant"""
        model = ConcreteModel()
        model.obj = Objective(expr=0.0)
        self.assertEqual(model.obj(), 0.0)
        self.assertEqual(value(model.obj), 0.0)
        self.assertEqual(value(model.obj._data[None]), 0.0)

    def test_mutable_param_expr(self):
        """Test expr option with a single mutable param"""
        model = ConcreteModel()
        model.p = Param(initialize=1.0, mutable=True)
        model.obj = Objective(expr=model.p)
        self.assertEqual(model.obj(), 1.0)
        self.assertEqual(value(model.obj), 1.0)
        self.assertEqual(value(model.obj._data[None]), 1.0)

    def test_immutable_param_expr(self):
        """Test expr option a single immutable param"""
        model = ConcreteModel()
        model.p = Param(initialize=1.0, mutable=False)
        model.obj = Objective(expr=model.p)
        self.assertEqual(model.obj(), 1.0)
        self.assertEqual(value(model.obj), 1.0)
        self.assertEqual(value(model.obj._data[None]), 1.0)

    def test_var_expr(self):
        """Test expr option with a single var"""
        model = ConcreteModel()
        model.x = Var(initialize=1.0)
        model.obj = Objective(expr=model.x)
        self.assertEqual(model.obj(), 1.0)
        self.assertEqual(value(model.obj), 1.0)
        self.assertEqual(value(model.obj._data[None]), 1.0)

    def test_expr1_option(self):
        """Test expr option"""
        model = ConcreteModel()
        model.B = RangeSet(1, 4)
        model.x = Var(model.B, initialize=2)
        ans = 0
        for i in model.B:
            ans = ans + model.x[i]
        model.obj = Objective(expr=ans)
        self.assertEqual(model.obj(), 8)
        self.assertEqual(value(model.obj), 8)
        self.assertEqual(value(model.obj._data[None]), 8)

    def test_expr2_option(self):
        """Test expr option"""
        model = ConcreteModel()
        model.x = Var(initialize=2)
        model.obj = Objective(expr=model.x)
        self.assertEqual(model.obj(), 2)
        self.assertEqual(value(model.obj), 2)
        self.assertEqual(value(model.obj._data[None]), 2)

    def test_rule_option(self):
        """Test rule option"""
        model = ConcreteModel()

        def f(model):
            ans = 0
            for i in [1, 2, 3, 4]:
                ans = ans + model.x[i]
            return ans
        model.x = Var(RangeSet(1, 4), initialize=2)
        model.obj = Objective(rule=f)
        self.assertEqual(model.obj(), 8)
        self.assertEqual(value(model.obj), 8)
        self.assertEqual(value(model.obj._data[None]), 8)

    def test_arguments(self):
        """Test that arguments notare of type ScalarSet"""
        model = ConcreteModel()

        def rule(model):
            return 1
        try:
            model.obj = Objective(model, rule=rule)
        except TypeError:
            pass
        else:
            self.fail('Objective should only accept ScalarSets')

    def test_sense_option(self):
        """Test sense option"""
        model = ConcreteModel()

        def rule(model):
            return 1.0
        model.obj = Objective(sense=maximize, rule=rule)
        self.assertEqual(model.obj.sense, maximize)
        self.assertEqual(model.obj.is_minimizing(), False)

    def test_dim(self):
        """Test dim method"""
        model = ConcreteModel()

        def rule(model):
            return 1
        model.obj = Objective(rule=rule)
        self.assertEqual(model.obj.dim(), 0)

    def test_keys(self):
        """Test keys method"""
        model = ConcreteModel()

        def rule(model):
            return 1
        model.obj = Objective(rule=rule)
        self.assertEqual(list(model.obj.keys()), [None])
        self.assertEqual(id(model.obj), id(model.obj[None]))

    def test_len(self):
        """Test len method"""
        model = AbstractModel()

        def rule(model):
            return 1.0
        model.obj = Objective(rule=rule)
        self.assertEqual(len(model.obj), 0)
        inst = model.create_instance()
        self.assertEqual(len(inst.obj), 1)
        model = AbstractModel()
        'Test rule option'

        def f(model):
            ans = 0
            for i in model.x.keys():
                ans = ans + model.x[i]
            return ans
        model = AbstractModel()
        model.x = Var(RangeSet(1, 4), initialize=2)
        model.obj = Objective(rule=f)
        self.assertEqual(len(model.obj), 0)
        inst = model.create_instance()
        self.assertEqual(len(inst.obj), 1)

    def test_keys_empty(self):
        """Test keys method"""
        model = ConcreteModel()
        model.o = Objective()
        self.assertEqual(list(model.o.keys()), [])

    def test_len_empty(self):
        """Test len method"""
        model = ConcreteModel()
        model.o = Objective()
        self.assertEqual(len(model.o), 0)