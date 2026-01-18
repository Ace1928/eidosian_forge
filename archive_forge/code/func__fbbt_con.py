from collections import defaultdict
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.contrib.fbbt.expression_bounds_walker import ExpressionBoundsVisitor
import pyomo.core.expr.numeric_expr as numeric_expr
from pyomo.core.expr.visitor import (
from pyomo.core.expr.numvalue import nonpyomo_leaf_types, value
from pyomo.core.expr.numvalue import is_fixed
import pyomo.contrib.fbbt.interval as interval
import math
from pyomo.core.base.block import Block
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var
from pyomo.gdp import Disjunct
from pyomo.core.base.expression import _GeneralExpressionData, ScalarExpression
import logging
from pyomo.common.errors import InfeasibleConstraintException, PyomoException
from pyomo.common.config import (
from pyomo.common.numeric_types import native_types
from the constraint, we know that 1 <= x*y + z <= 1, so we may 
def _fbbt_con(con, config):
    """
    Feasibility based bounds tightening for a constraint. This function attempts to improve the bounds of each variable
    in the constraint based on the bounds of the constraint and the bounds of the other variables in the constraint.
    For example:

    >>> import pyomo.environ as pe
    >>> from pyomo.contrib.fbbt.fbbt import fbbt
    >>> m = pe.ConcreteModel()
    >>> m.x = pe.Var(bounds=(-1,1))
    >>> m.y = pe.Var(bounds=(-2,2))
    >>> m.z = pe.Var()
    >>> m.c = pe.Constraint(expr=m.x*m.y + m.z == 1)
    >>> fbbt(m.c)
    >>> print(m.z.lb, m.z.ub)
    -1.0 3.0

    Parameters
    ----------
    con: pyomo.core.base.constraint.Constraint
        constraint on which to perform fbbt
    config: ConfigDict
        see documentation for fbbt

    Returns
    -------
    new_var_bounds: ComponentMap
        A ComponentMap mapping from variables a tuple containing the lower and upper bounds, respectively, computed
        from FBBT.
    """
    if not con.active:
        return ComponentMap()
    bnds_dict = ComponentMap()
    visitorA = _FBBTVisitorLeafToRoot(bnds_dict, feasibility_tol=config.feasibility_tol)
    visitorA.walk_expression(con.body)
    _lb = value(con.lower)
    _ub = value(con.upper)
    if _lb is None:
        _lb = -interval.inf
    if _ub is None:
        _ub = interval.inf
    lb, ub = bnds_dict[con.body]
    if lb > _ub + config.feasibility_tol or ub < _lb - config.feasibility_tol:
        raise InfeasibleConstraintException('Detected an infeasible constraint during FBBT: {0}'.format(str(con)))
    if config.deactivate_satisfied_constraints:
        if lb >= _lb - config.feasibility_tol and ub <= _ub + config.feasibility_tol:
            con.deactivate()
    if _lb > lb:
        lb = _lb
    if _ub < ub:
        ub = _ub
    bnds_dict[con.body] = (lb, ub)
    visitorB = _FBBTVisitorRootToLeaf(bnds_dict, integer_tol=config.integer_tol, feasibility_tol=config.feasibility_tol)
    visitorB.dfs_postorder_stack(con.body)
    new_var_bounds = ComponentMap()
    for _node, _bnds in bnds_dict.items():
        if _node.__class__ in nonpyomo_leaf_types:
            continue
        if _node.is_variable_type():
            lb, ub = bnds_dict[_node]
            if lb == -interval.inf:
                lb = None
            if ub == interval.inf:
                ub = None
            new_var_bounds[_node] = (lb, ub)
    return new_var_bounds