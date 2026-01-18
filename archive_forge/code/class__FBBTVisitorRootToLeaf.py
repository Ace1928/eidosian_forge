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
class _FBBTVisitorRootToLeaf(ExpressionValueVisitor):
    """
    This walker propagates bounds from the constraint back to the
    variables. Note that the bounds on every node in the tree must
    first be computed with _FBBTVisitorLeafToRoot.
    """

    def __init__(self, bnds_dict, integer_tol=0.0001, feasibility_tol=1e-08):
        """
        Parameters
        ----------
        bnds_dict: ComponentMap
        integer_tol: float
        feasibility_tol: float
            If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
            feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
            is also used when performing certain interval arithmetic operations to ensure that none of the feasible
            region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
            is more conservative).
        """
        self.bnds_dict = bnds_dict
        self.integer_tol = integer_tol
        self.feasibility_tol = feasibility_tol

    def visit(self, node, values):
        pass

    def visiting_potential_leaf(self, node):
        if node.__class__ in nonpyomo_leaf_types:
            lb, ub = self.bnds_dict[node]
            if abs(lb - value(node)) > self.feasibility_tol:
                raise InfeasibleConstraintException('Detected an infeasible constraint.')
            if abs(ub - value(node)) > self.feasibility_tol:
                raise InfeasibleConstraintException('Detected an infeasible constraint.')
            return (True, None)
        if node.is_variable_type():
            lb, ub = self.bnds_dict[node]
            lb, ub = self.bnds_dict[node]
            if lb > ub:
                if lb - self.feasibility_tol > ub:
                    raise InfeasibleConstraintException('Lower bound ({1}) computed for variable {0} is larger than the computed upper bound ({2}).'.format(node, lb, ub))
                else:
                    '\n                    If we reach this code, then lb > ub, but not by more than feasibility_tol.\n                    Now we want to decrease lb slightly and increase ub slightly so that lb <= ub.\n                    However, we also have to make sure we do not make lb lower than the original lower bound\n                    and make sure we do not make ub larger than the original upper bound. This is what\n                    _check_and_reset_bounds is for.\n                    '
                    lb -= self.feasibility_tol
                    ub += self.feasibility_tol
                    lb, ub = _check_and_reset_bounds(node, lb, ub)
                    self.bnds_dict[node] = (lb, ub)
            if lb == interval.inf:
                raise InfeasibleConstraintException('Computed a lower bound of +inf for variable {0}'.format(node))
            if ub == -interval.inf:
                raise InfeasibleConstraintException('Computed an upper bound of -inf for variable {0}'.format(node))
            if node.is_binary() or node.is_integer():
                '\n                This bit of code has two purposes:\n                1) Improve the bounds on binary and integer variables with the fact that they are integer.\n                2) Account for roundoff error. If the lower bound of a binary variable comes back as\n                   1e-16, the lower bound may actually be 0. This could potentially cause problems when\n                   handing the problem to a MIP solver. Some solvers are robust to this, but some may not be\n                   and may give the wrong solution. Even if the correct solution is found, this could\n                   introduce numerical problems.\n                '
                if lb > -interval.inf:
                    lb = max(math.floor(lb), math.ceil(lb - self.integer_tol))
                if ub < interval.inf:
                    ub = min(math.ceil(ub), math.floor(ub + self.integer_tol))
                '\n                We have to make sure we do not make lb lower than the original lower bound\n                and make sure we do not make ub larger than the original upper bound. This is what \n                _check_and_reset_bounds is for.\n                '
                lb, ub = _check_and_reset_bounds(node, lb, ub)
                self.bnds_dict[node] = (lb, ub)
            if lb != -interval.inf:
                node.setlb(lb)
            if ub != interval.inf:
                node.setub(ub)
            return (True, None)
        if not node.is_potentially_variable():
            lb, ub = self.bnds_dict[node]
            if abs(lb - value(node)) > self.feasibility_tol:
                raise InfeasibleConstraintException('Detected an infeasible constraint.')
            if abs(ub - value(node)) > self.feasibility_tol:
                raise InfeasibleConstraintException('Detected an infeasible constraint.')
            return (True, None)
        if node.__class__ is numeric_expr.ExternalFunctionExpression:
            return (True, None)
        if node.__class__ in _prop_bnds_root_to_leaf_map:
            _prop_bnds_root_to_leaf_map[node.__class__](node, self.bnds_dict, self.feasibility_tol)
        else:
            logger.warning('Unsupported expression type for FBBT: {0}. Bounds will not be improved in this part of the tree.'.format(str(type(node))))
        return (False, None)