import logging
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.common.modeling import NOTSET
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.visitor import ExpressionValueVisitor
import pyomo.core.expr as EXPR
def _rel_diff(self, a, b):
    scale = min(abs(a), abs(b))
    if scale < 1.0:
        scale = 1.0
    return abs(a - b) / scale