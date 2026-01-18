import logging
import sys
from operator import itemgetter
from itertools import filterfalse
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.numeric_types import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.relational_expr import (
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, _EvaluationVisitor
from pyomo.core.expr import is_fixed, value
from pyomo.core.base.expression import Expression
import pyomo.core.kernel as kernel
from pyomo.repn.util import (
def _initialize_exit_node_dispatcher(exit_handlers):
    exit_dispatcher = {}
    for cls, handlers in exit_handlers.items():
        for args, fcn in handlers.items():
            exit_dispatcher[cls, *args] = fcn
    return exit_dispatcher