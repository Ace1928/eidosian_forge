import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.opt import check_available_solvers
from pyomo.environ import (
@unittest.skipIf(not 'glpk' in solvers, 'glpk solver is not available')
def Xtest_pyomo_Set_dat_file_domain(self):
    self.model = AbstractModel()
    self.model.s = Set()
    self.model.y = Var([1, 2], within=self.model.s)

    def obj_rule(model):
        return sum((model.y[i] * (-1) ** (i - 1) for i in model.y))
    self.model.obj = Objective(rule=obj_rule)
    self.model.con = Constraint([1, 2], rule=lambda model, i: model.y[i] * (-1) ** (i - 1) >= 1.1 ** (2 - i) * (-2.9) ** (i - 1))
    self.instance = self.model.create_instance(currdir + 'vars_dat_file.dat')
    self.opt = SolverFactory('glpk')
    self.results = self.opt.solve(self.instance)
    self.instance.load(self.results)
    self.assertEqual(self.instance.y[1], 2)
    self.assertEqual(self.instance.y[2], 2)