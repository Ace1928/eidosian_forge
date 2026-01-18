import logging
from math import pi
from pyomo.common.collections import ComponentMap
from pyomo.contrib.fbbt.interval import (
from pyomo.core.base.expression import Expression
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.logical_expr import BooleanExpression
from pyomo.core.expr.relational_expr import (
from pyomo.core.expr.numvalue import native_numeric_types, native_types, value
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.repn.util import BeforeChildDispatcher, ExitNodeDispatcher
def _handle_asin(visitor, node, arg):
    return asin(*arg, -pi / 2, pi / 2, visitor.feasibility_tol)