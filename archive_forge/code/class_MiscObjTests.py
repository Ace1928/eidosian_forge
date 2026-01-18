import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
class MiscObjTests(unittest.TestCase):

    def test_constructor(self):
        a = Objective(name='b')
        self.assertEqual(a.local_name, 'b')
        try:
            a = Objective(foo='bar')
            self.fail("Can't specify an unexpected constructor option")
        except ValueError:
            pass

    def test_rule(self):

        def rule1(model):
            return []
        model = ConcreteModel()
        try:
            model.o = Objective(rule=rule1)
            self.fail('Error generating objective')
        except Exception:
            pass
        model = ConcreteModel()

        def rule1(model):
            return 1.1
        model = ConcreteModel()
        model.o = Objective(rule=rule1)
        self.assertEqual(model.o(), 1.1)
        model = ConcreteModel()

        def rule1(model, i):
            return 1.1
        model = ConcreteModel()
        model.a = Set(initialize=[1, 2, 3])
        try:
            model.o = Objective(model.a, rule=rule1)
        except Exception:
            self.fail('Error generating objective')

    def test_abstract_index(self):
        model = AbstractModel()
        model.A = Set()
        model.B = Set()
        model.C = model.A | model.B
        model.x = Objective(model.C)