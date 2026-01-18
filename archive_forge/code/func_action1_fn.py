from io import StringIO
import os
import pyomo.common.unittest as unittest
from pyomo.environ import AbstractModel, Param, Set, BuildAction, value
def action1_fn(model):
    model.A = 4.3