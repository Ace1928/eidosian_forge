import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.core.base import IntegerSet
from pyomo.core.expr.numeric_expr import (
from pyomo.core.staleflag import StaleFlagManager
from pyomo.environ import (
from pyomo.core.base.units_container import units, pint_available, UnitsError
def Xtest_setlb_nondata_expression(self):
    model = ConcreteModel()
    model.x = Var()
    model.e = Expression()
    with self.assertRaises(ValueError):
        model.x.setlb(model.e)
    model.e.expr = 1.0
    with self.assertRaises(ValueError):
        model.x.setlb(model.e)
    model.y = Var()
    with self.assertRaises(ValueError):
        model.x.setlb(model.y)
    model.y.value = 1.0
    with self.assertRaises(ValueError):
        model.x.setlb(model.y)
    model.y.fix()
    with self.assertRaises(ValueError):
        model.x.setlb(model.y + 1)