import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import AbstractModel, Param, Var, Constraint
def X_bounds_rule(model):
    return (model.A * (model.B - model.C), model.A * (model.B + model.C))