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
class ExpressionBoundsExitNodeDispatcher(ExitNodeDispatcher):

    def unexpected_expression_type(self, visitor, node, *args):
        if isinstance(node, NumericExpression):
            ans = (-inf, inf)
        elif isinstance(node, BooleanExpression):
            ans = (BoolFlag(False), BoolFlag(True))
        else:
            super().unexpected_expression_type(visitor, node, *args)
        logger.warning(f"Unexpected expression node type '{type(node).__name__}' found while walking expression tree; returning {ans} for the expression bounds.")
        return ans