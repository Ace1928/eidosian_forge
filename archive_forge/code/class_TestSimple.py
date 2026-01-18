import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, AbstractModel, SOSConstraint, Var, Set
class TestSimple(unittest.TestCase):

    def setUp(self):
        self.M = ConcreteModel()

    def tearDown(self):
        self.M = None

    def test_num_vars(self):
        self.M.x = Var([1, 2, 3])
        self.M.c = SOSConstraint(var=self.M.x, sos=1)
        self.assertEqual(self.M.c.num_variables(), 3)

    def test_level(self):
        self.M.x = Var([1, 2, 3])
        self.M.c = SOSConstraint(var=self.M.x, sos=1)
        self.assertEqual(self.M.c.level, 1)
        self.M.c.level = 2
        self.assertEqual(self.M.c.level, 2)
        try:
            self.M.c.level = -1
            self.fail('Expected ValueError')
        except ValueError:
            pass

    def test_get_variables(self):
        self.M.x = Var([1, 2, 3])
        self.M.c = SOSConstraint(var=self.M.x, sos=1)
        self.assertEqual(set((id(v) for v in self.M.c.get_variables())), set((id(v) for v in self.M.x.values())))