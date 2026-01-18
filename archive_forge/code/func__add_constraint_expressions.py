from pyomo.gdp import GDP_Error
from pyomo.common.collections import ComponentSet
from pyomo.contrib.fbbt.expression_bounds_walker import ExpressionBoundsVisitor
import pyomo.contrib.fbbt.interval as interval
from pyomo.core import Suffix
def _add_constraint_expressions(self, c, i, M, indicator_var, newConstraint, constraintMap):
    unique = len(newConstraint)
    name = c.local_name + '_%s' % unique
    if c.lower is not None:
        if M[0] is None:
            raise GDP_Error("Cannot relax disjunctive constraint '%s' because M is not defined." % name)
        M_expr = M[0] * (1 - indicator_var)
        newConstraint.add((name, i, 'lb'), c.lower <= c.body - M_expr)
        constraintMap['transformedConstraints'][c] = [newConstraint[name, i, 'lb']]
        constraintMap['srcConstraints'][newConstraint[name, i, 'lb']] = c
    if c.upper is not None:
        if M[1] is None:
            raise GDP_Error("Cannot relax disjunctive constraint '%s' because M is not defined." % name)
        M_expr = M[1] * (1 - indicator_var)
        newConstraint.add((name, i, 'ub'), c.body - M_expr <= c.upper)
        transformed = constraintMap['transformedConstraints'].get(c)
        if transformed is not None:
            constraintMap['transformedConstraints'][c].append(newConstraint[name, i, 'ub'])
        else:
            constraintMap['transformedConstraints'][c] = [newConstraint[name, i, 'ub']]
        constraintMap['srcConstraints'][newConstraint[name, i, 'ub']] = c