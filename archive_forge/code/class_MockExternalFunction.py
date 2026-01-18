import logging
import math
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr.numvalue import is_fixed
from pyomo.core.expr.compare import assertExpressionsStructurallyEqual
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.environ import ConcreteModel, Param, Var, ExternalFunction
class MockExternalFunction(object):

    def evaluate(self, args):
        x, = args
        return (math.log(x) / math.log(2)) ** 2

    def getname(self):
        return 'mock_fcn'