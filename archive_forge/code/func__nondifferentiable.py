import operator
import sys
from pyomo.common import DeveloperError
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import NondifferentiableError
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import value, native_types
def _nondifferentiable(x):
    if type(x[1]) is tuple:
        wrt = x[1][0]
    else:
        wrt = x[1]
    raise NondifferentiableError("The sub-expression '%s' is not differentiable with respect to %s" % (x[0], wrt))