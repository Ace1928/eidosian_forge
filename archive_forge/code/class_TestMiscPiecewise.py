import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import AbstractModel, ConcreteModel, Set, Var, Piecewise, Constraint
class TestMiscPiecewise(unittest.TestCase):

    def test_activate_deactivate_indexed(self):
        model = ConcreteModel()
        model.s = Set(initialize=[1])
        model.y = Var(model.s)
        model.x = Var(model.s, bounds=(-1, 1))
        args = ([1], model.y, model.x)
        keywords = {'pw_pts': {1: [-1, 0, 1]}, 'pw_constr_type': 'EQ', 'f_rule': lambda model, i, x: x ** 2}
        model.c = Piecewise(*args, **keywords)
        self.assertTrue(len(model.c[1].component_map(Constraint)) > 0)
        self.assertTrue(len(model.c[1].component_map(Constraint, active=True)) > 0)
        self.assertEqual(model.c.active, True)
        self.assertEqual(model.c[1].active, True)
        model.c[1].deactivate()
        self.assertTrue(len(model.c[1].component_map(Constraint)) > 0)
        self.assertTrue(len(model.c[1].component_map(Constraint, active=True)) > 0)
        self.assertEqual(model.c.active, True)
        self.assertEqual(model.c[1].active, False)
        model.c[1].activate()
        self.assertTrue(len(model.c[1].component_map(Constraint)) > 0)
        self.assertTrue(len(model.c[1].component_map(Constraint, active=True)) > 0)
        self.assertEqual(model.c.active, True)
        self.assertEqual(model.c[1].active, True)
        model.c.deactivate()
        self.assertTrue(len(model.c[1].component_map(Constraint)) > 0)
        self.assertTrue(len(model.c[1].component_map(Constraint, active=True)) > 0)
        self.assertEqual(model.c.active, False)
        self.assertEqual(model.c[1].active, False)

    def test_activate_deactivate_nonindexed(self):
        model = ConcreteModel()
        model.y = Var()
        model.x = Var(bounds=(-1, 1))
        args = (model.y, model.x)
        keywords = {'pw_pts': [-1, 0, 1], 'pw_constr_type': 'EQ', 'f_rule': lambda model, x: x ** 2}
        model.c = Piecewise(*args, **keywords)
        self.assertTrue(len(model.c.component_map(Constraint)) > 0)
        self.assertTrue(len(model.c.component_map(Constraint, active=True)) > 0)
        self.assertEqual(model.c.active, True)
        model.c.deactivate()
        self.assertTrue(len(model.c.component_map(Constraint)) > 0)
        self.assertTrue(len(model.c.component_map(Constraint, active=True)) > 0)
        self.assertEqual(model.c.active, False)
        model.c.activate()
        self.assertTrue(len(model.c.component_map(Constraint)) > 0)
        self.assertTrue(len(model.c.component_map(Constraint, active=True)) > 0)
        self.assertEqual(model.c.active, True)

    def test_indexed_with_nonindexed_vars(self):
        model = ConcreteModel()
        model.range1 = Var()
        model.x = Var(bounds=(-1, 1))
        args = ([1], model.range1, model.x)
        keywords = {'pw_pts': {1: [-1, 0, 1]}, 'pw_constr_type': 'EQ', 'f_rule': lambda model, i, x: x ** 2}
        model.con1 = Piecewise(*args, **keywords)
        model.range2 = Var([1])
        model.y = Var([1], bounds=(-1, 1))
        args = ([1], model.range2, model.y)
        model.con2 = Piecewise(*args, **keywords)
        args = ([1], model.range2, model.y[1])
        model.con3 = Piecewise(*args, **keywords)

    def test_nonindexed_with_indexed_vars(self):
        model = ConcreteModel()
        model.range = Var([1])
        model.x = Var([1], bounds=(-1, 1))
        args = (model.range[1], model.x[1])
        keywords = {'pw_pts': [-1, 0, 1], 'pw_constr_type': 'EQ', 'f_rule': lambda model, x: x ** 2}
        model.con1 = Piecewise(*args, **keywords)

    def test_abstract_piecewise(self):
        model = AbstractModel()
        model.range = Var()
        model.x = Var(bounds=(-1, 1))
        args = (model.range, model.x)
        keywords = {'pw_pts': [-1, 0, 1], 'pw_constr_type': 'EQ', 'f_rule': lambda model, x: x ** 2}
        model.con = Piecewise(*args, **keywords)
        instance = model.create_instance()

    def test_concrete_piecewise(self):
        model = ConcreteModel()
        model.range = Var()
        model.x = Var(bounds=(-1, 1))
        args = (model.range, model.x)
        keywords = {'pw_pts': [-1, 0, 1], 'pw_constr_type': 'EQ', 'f_rule': lambda model, x: x ** 2}
        model.con = Piecewise(*args, **keywords)