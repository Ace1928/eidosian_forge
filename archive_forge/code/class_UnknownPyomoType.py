import math
import pickle
from pyomo.common.errors import PyomoException
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.log import LoggingIntercept
from pyomo.util.check_units import assert_units_consistent, check_units_equivalent
from pyomo.core.expr import inequality
from pyomo.core.expr.numvalue import NumericConstant
import pyomo.core.expr as EXPR
from pyomo.core.base.units_container import (
from io import StringIO
class UnknownPyomoType(object):

    def is_expression_type(self, expression_system=None):
        return False

    def is_numeric_type(self):
        return False

    def is_logical_type(self):
        return False