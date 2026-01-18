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
def Xtest_setlb_data_expression(self):
    model = ConcreteModel()
    model.x = Var()
    model.p = Param(mutable=True)
    model.x.setlb(model.p)
    model.x.setlb(model.p ** 2 + 1)
    model.p.value = 1.0
    model.x.setlb(model.p)
    model.x.setlb(model.p ** 2)
    model.x.setlb(1.0)