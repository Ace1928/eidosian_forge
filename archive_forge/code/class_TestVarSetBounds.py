import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.opt import check_available_solvers
from pyomo.environ import (
class TestVarSetBounds(unittest.TestCase):

    @unittest.skipIf(not 'glpk' in solvers, 'glpk solver is not available')
    def Xtest_rangeset_domain(self):
        self.model = ConcreteModel()
        self.model.s = RangeSet(3)
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

    def Xtest_pyomo_Set_domain_empty(self):
        with self.assertRaises(ValueError) as cm:
            self.model = ConcreteModel()
            self.model.s = Set(initialize=[])
            self.model.y = Var([1, 2], within=self.model.s)

    def Xtest_pyomo_Set_domain_missing(self):
        with self.assertRaises(ValueError) as cm:
            self.model = ConcreteModel()
            self.model.s = Set(initialize=[1, 4, 5])
            self.model.y = Var([1, 2], within=self.model.s)

    def Xtest_pyomo_Set_domain_duplicates(self):
        with self.assertRaises(ValueError) as cm:
            self.model = ConcreteModel()
            self.model.s = Set(initialize=[1.7, 2, 3])
            self.model.y = Var([1, 2], within=self.model.s)

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

    def Xtest_pyomo_Set_dat_file_domain_empty(self):
        with self.assertRaises(ValueError) as cm:
            self.model = AbstractModel()
            self.model.s = Set()
            self.model.y = Var([1, 2], within=self.model.s)
            self.instance = self.model.create_instance(currdir + 'vars_dat_file_empty.dat')

    def Xtest_pyomo_Set_dat_file_domain_missing(self):
        with self.assertRaises(ValueError) as cm:
            self.model = AbstractModel()
            self.model.s = Set()
            self.model.y = Var([1, 2], within=self.model.s)
            self.instance = self.model.create_instance(currdir + 'vars_dat_file_missing.dat')

    def Xtest_pyomo_Set_dat_file_domain_duplicates(self):
        with self.assertRaises(ValueError) as cm:
            self.model = AbstractModel()
            self.model.s = Set()
            self.model.y = Var([1, 2], within=self.model.s)
            self.instance = self.model.create_instance(currdir + 'vars_dat_file_nonint.dat')

    @unittest.skipIf(not 'glpk' in solvers, 'glpk solver is not available')
    def Xtest_list_domain(self):
        self.model = ConcreteModel()
        self.model.y = Var([1, 2], within=[1, 2, 3])
        self.model.obj = Objective(expr=self.model.y[1] - self.model.y[2])
        self.model.con1 = Constraint(expr=self.model.y[1] >= 1.1)
        self.model.con2 = Constraint(expr=self.model.y[2] <= 2.9)
        self.instance = self.model.create_instance()
        self.opt = solver['glpk']
        self.results = self.opt.solve(self.instance)
        self.instance.load(self.results)
        self.assertEqual(self.instance.y[1], 2)
        self.assertEqual(self.instance.y[2], 2)

    def Xtest_list_domain_empty(self):
        with self.assertRaises(ValueError) as cm:
            self.model = ConcreteModel()
            self.model.y = Var([1, 2], within=[])

    def Xtest_list_domain_bad_missing(self):
        with self.assertRaises(ValueError) as cm:
            self.model = ConcreteModel()
            self.model.y = Var([1, 2], within=[1, 4, 5])

    def Xtest_list_domain_bad_duplicates(self):
        with self.assertRaises(ValueError) as cm:
            self.model = ConcreteModel()
            self.model.y = Var([1, 2], within=[1, 1, 2, 3])

    def Xtest_list_domain_bad_duplicates(self):
        with self.assertRaises(ValueError) as cm:
            self.model = ConcreteModel()
            self.model.y = Var([1, 2], within=[1.7, 2, 3])

    @unittest.skipIf(not 'glpk' in solvers, 'glpk solver is not available')
    def Xtest_set_domain(self):
        self.model = ConcreteModel()
        self.model.y = Var([1, 2], within=set([1, 2, 3]))
        self.model.obj = Objective(expr=self.model.y[1] - self.model.y[2])
        self.model.con1 = Constraint(expr=self.model.y[1] >= 1.1)
        self.model.con2 = Constraint(expr=self.model.y[2] <= 2.9)
        self.instance = self.model.create_instance()
        self.opt = solver['glpk']
        self.results = self.opt.solve(self.instance)
        self.instance.load(self.results)
        self.assertEqual(self.instance.y[1], 2)
        self.assertEqual(self.instance.y[2], 2)

    def Xtest_set_domain_empty(self):
        with self.assertRaises(ValueError) as cm:
            self.model = ConcreteModel()
            self.model.y = Var([2, 2], within=set([]))

    def Xtest_set_domain_bad_missing(self):
        with self.assertRaises(ValueError) as cm:
            self.model = ConcreteModel()
            self.model.y = Var([1, 2], within=set([1, 4, 5]))

    def Xtest_set_domain_bad_duplicates(self):
        with self.assertRaises(ValueError) as cm:
            self.model = ConcreteModel()
            self.model.y = Var([1, 2], within=set([1, 1, 2, 3]))

    def Xtest_set_domain_bad_duplicates(self):
        with self.assertRaises(ValueError) as cm:
            self.model = ConcreteModel()
            self.model.y = Var([1, 2], within=set([1.7, 2, 3]))

    @unittest.skipIf(not 'glpk' in solvers, 'glpk solver is not available')
    def Xtest_rangeset_domain(self):
        self.model = ConcreteModel()
        self.model.y = Var([1, 2], within=range(4))
        self.model.obj = Objective(expr=self.model.y[1] - self.model.y[2])
        self.model.con1 = Constraint(expr=self.model.y[1] >= 1.1)
        self.model.con2 = Constraint(expr=self.model.y[2] <= 2.9)
        self.instance = self.model.create_instance()
        self.opt = solver['glpk']
        self.results = self.opt.solve(self.instance)
        self.instance.load(self.results)
        self.assertEqual(self.instance.y[1], 2)
        self.assertEqual(self.instance.y[2], 2)