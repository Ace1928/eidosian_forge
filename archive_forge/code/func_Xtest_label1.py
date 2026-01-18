import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import AbstractModel, Set, Param, Var, Objective
def Xtest_label1(self):
    model = AbstractModel()
    model.A = Set(initialize=[1, 2, 3])
    model.B = Param(model.A, initialize={1: 100, 2: 200, 3: 300})
    model.x = Var(model.A)
    model.y = Var(model.A)
    instance = model.create_instance()
    instance.preprocess()
    self.assertEqual(instance.num_used_variables(), 0)