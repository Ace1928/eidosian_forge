import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.opt import check_available_solvers
from pyomo.environ import (
@unittest.skipIf(not 'glpk' in solvers, 'glpk solver is not available')
def Xtest_pyomo_Set_domain(self):
    self.model = ConcreteModel()
    self.model.s = Set(initialize=[1, 2, 3])
    self.model.y = Var([1, 2], within=self.model.s)
    self.model.obj = Objective(expr=self.model.y[1] - self.model.y[2])
    self.model.con1 = Constraint(expr=self.model.y[1] >= 1.1)
    self.model.con2 = Constraint(expr=self.model.y[2] <= 2.9)
    self.instance = self.model.create_instance()
    self.opt = SolverFactory('glpk')
    self.results = self.opt.solve(self.instance)
    self.instance.load(self.results)
    self.assertEqual(self.instance.y[1], 2)
    self.assertEqual(self.instance.y[2], 2)