import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.opt import check_available_solvers
from pyomo.environ import (
def Xtest_pyomo_Set_domain_duplicates(self):
    with self.assertRaises(ValueError) as cm:
        self.model = ConcreteModel()
        self.model.s = Set(initialize=[1.7, 2, 3])
        self.model.y = Var([1, 2], within=self.model.s)