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
class ExpressionBoundsVisitor(StreamBasedExpressionVisitor):
    """
    Walker to calculate bounds on an expression, from leaf to root, with
    caching of terminal node bounds (Vars and Expressions)

    NOTE: If anything changes on the model (e.g., Var bounds, fixing, mutable
    Param values, etc), then you need to either create a new instance of this
    walker, or clear self.leaf_bounds!

    Parameters
    ----------
    leaf_bounds: ComponentMap in which to cache bounds at leaves of the expression
        tree
    feasibility_tol: float, feasibility tolerance for interval arithmetic
        calculations
    use_fixed_var_values_as_bounds: bool, whether or not to use the values of
        fixed Vars as the upper and lower bounds for those Vars or to instead
        ignore fixed status and use the bounds. Set to 'True' if you do not
        anticipate the fixed status of Variables to change for the duration that
        the computed bounds should be valid.
    """
    _before_child_handlers = ExpressionBoundsBeforeChildDispatcher()
    _operator_dispatcher = ExpressionBoundsExitNodeDispatcher({ProductExpression: _handle_ProductExpression, DivisionExpression: _handle_DivisionExpression, PowExpression: _handle_PowExpression, AbsExpression: _handle_AbsExpression, SumExpression: _handle_SumExpression, MonomialTermExpression: _handle_ProductExpression, NegationExpression: _handle_NegationExpression, UnaryFunctionExpression: _handle_UnaryFunctionExpression, LinearExpression: _handle_SumExpression, Expression: _handle_named_expression, ExternalFunctionExpression: _handle_unknowable_bounds, EqualityExpression: _handle_equality, InequalityExpression: _handle_inequality, RangedExpression: _handle_ranged, Expr_ifExpression: _handle_expr_if})

    def __init__(self, leaf_bounds=None, feasibility_tol=1e-08, use_fixed_var_values_as_bounds=False):
        super().__init__()
        self.leaf_bounds = leaf_bounds if leaf_bounds is not None else ComponentMap()
        self.feasibility_tol = feasibility_tol
        self.use_fixed_var_values_as_bounds = use_fixed_var_values_as_bounds

    def initializeWalker(self, expr):
        walk, result = self.beforeChild(None, expr, 0)
        if not walk:
            return (False, result)
        return (True, expr)

    def beforeChild(self, node, child, child_idx):
        return self._before_child_handlers[child.__class__](self, child)

    def exitNode(self, node, data):
        return self._operator_dispatcher[node.__class__](self, node, *data)