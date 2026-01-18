import logging
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.common.modeling import NOTSET
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.visitor import ExpressionValueVisitor
import pyomo.core.expr as EXPR
class UnitsError(Exception):
    """
    An exception class for all general errors/warnings associated with units
    """

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return str(self.msg)