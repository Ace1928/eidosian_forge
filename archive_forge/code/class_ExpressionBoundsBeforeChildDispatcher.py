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
class ExpressionBoundsBeforeChildDispatcher(BeforeChildDispatcher):
    __slots__ = ()

    def __init__(self):
        self[ExternalFunctionExpression] = self._before_external_function

    @staticmethod
    def _before_external_function(visitor, child):
        return (False, (-inf, inf))

    @staticmethod
    def _before_native_numeric(visitor, child):
        return (False, (child, child))

    @staticmethod
    def _before_native_logical(visitor, child):
        return (False, (BoolFlag(child), BoolFlag(child)))

    @staticmethod
    def _before_var(visitor, child):
        leaf_bounds = visitor.leaf_bounds
        if child in leaf_bounds:
            pass
        elif child.is_fixed() and visitor.use_fixed_var_values_as_bounds:
            val = child.value
            try:
                ans = visitor._before_child_handlers[val.__class__](visitor, val)
            except ValueError:
                raise ValueError("Var '%s' is fixed to None. This value cannot be used to calculate bounds." % child.name) from None
            leaf_bounds[child] = ans[1]
            return ans
        else:
            lb = child.lb
            ub = child.ub
            if lb is None:
                lb = -inf
            if ub is None:
                ub = inf
            leaf_bounds[child] = (lb, ub)
        return (False, leaf_bounds[child])

    @staticmethod
    def _before_named_expression(visitor, child):
        leaf_bounds = visitor.leaf_bounds
        if child in leaf_bounds:
            return (False, leaf_bounds[child])
        else:
            return (True, None)

    @staticmethod
    def _before_param(visitor, child):
        val = child.value
        return visitor._before_child_handlers[val.__class__](visitor, val)

    @staticmethod
    def _before_string(visitor, child):
        raise ValueError(f'{child!r} ({type(child).__name__}) is not a valid numeric type. Cannot compute bounds on expression.')

    @staticmethod
    def _before_invalid(visitor, child):
        raise ValueError(f'{child!r} ({type(child).__name__}) is not a valid numeric type. Cannot compute bounds on expression.')

    @staticmethod
    def _before_complex(visitor, child):
        raise ValueError(f'Cannot compute bounds on expressions containing complex numbers. Encountered when processing {child}')

    @staticmethod
    def _before_npv(visitor, child):
        val = value(child)
        return visitor._before_child_handlers[val.__class__](visitor, val)