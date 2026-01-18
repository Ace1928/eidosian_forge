import logging
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.common.modeling import NOTSET
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.visitor import ExpressionValueVisitor
import pyomo.core.expr as EXPR
def _handle_external(self, node, values):
    ans = node._apply_operation([val.magnitude if val.__class__ is units._pint_registry.Quantity else val for val in values])
    unit = node.get_units()
    if unit is not None:
        ans = ans * unit._pint_unit
    return ans