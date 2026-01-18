import sys
import logging
import itertools
from pyomo.common.numeric_types import native_types, native_numeric_types
from pyomo.core.base import Constraint, Objective, ComponentMap
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import NumericConstant
from pyomo.core.base.objective import _GeneralObjectiveData, ScalarObjective
from pyomo.core.base import _ExpressionData, Expression
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.var import ScalarVar, Var, _GeneralVarData, value
from pyomo.core.base.param import ScalarParam, _ParamData
from pyomo.core.kernel.expression import expression, noclone
from pyomo.core.kernel.variable import IVariable, variable
from pyomo.core.kernel.objective import objective
from io import StringIO
def _collect_var(exp, multiplier, idMap, compute_values, verbose, quadratic):
    ans = Results()
    if exp.fixed:
        if compute_values:
            ans.constant += multiplier * value(exp)
        else:
            ans.constant += multiplier * exp
    else:
        id_ = id(exp)
        if id_ in idMap[None]:
            key = idMap[None][id_]
        else:
            key = len(idMap) - 1
            idMap[None][id_] = key
            idMap[key] = exp
        ans.linear[key] = multiplier
    return ans