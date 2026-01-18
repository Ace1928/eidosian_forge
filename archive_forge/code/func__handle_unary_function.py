import logging
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.common.modeling import NOTSET
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.visitor import ExpressionValueVisitor
import pyomo.core.expr as EXPR
def _handle_unary_function(self, node, values):
    ans = node._apply_operation(values)
    if node.getname() in self._unary_inverse_trig:
        ans = ans * units._pint_registry.radian
    return ans