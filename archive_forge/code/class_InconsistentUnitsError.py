import logging
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.common.modeling import NOTSET
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.visitor import ExpressionValueVisitor
import pyomo.core.expr as EXPR
class InconsistentUnitsError(UnitsError):
    """
    An exception indicating that inconsistent units are present on an expression.

    E.g., x == y, where x is in units of kg and y is in units of meter
    """

    def __init__(self, exp1, exp2, msg):
        msg = f'{msg}: {exp1} not compatible with {exp2}.'
        super(InconsistentUnitsError, self).__init__(msg)