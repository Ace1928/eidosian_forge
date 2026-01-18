from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.expr.numeric_expr import (
from pyomo.repn.quadratic import QuadraticRepnVisitor
from pyomo.repn.util import InvalidNumber
from pyomo.environ import ConcreteModel, Var, Param, Any, log
class VisitorConfig(object):

    def __init__(self):
        self.subexpr = {}
        self.var_map = {}
        self.var_order = {}
        self.sorter = None

    def __iter__(self):
        return iter((self.subexpr, self.var_map, self.var_order, self.sorter))