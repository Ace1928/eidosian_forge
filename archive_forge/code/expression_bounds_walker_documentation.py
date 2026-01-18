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
    