import sys
import logging
from pyomo.common.deprecation import (
from pyomo.core.expr.expr_common import ExpressionType
from pyomo.core.expr.numeric_expr import NumericValue
from pyomo.common.numeric_types import (
from pyomo.core.pyomoobject import PyomoObject
class NonNumericValue(object):
    """An object that contains a non-numeric value

    Constructor Arguments:
        value           The initial value.
    """
    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)