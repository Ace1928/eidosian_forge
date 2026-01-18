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
def _collect_standard_repn(exp, multiplier, idMap, compute_values, verbose, quadratic):
    fn = _repn_collectors.get(exp.__class__, None)
    if fn is not None:
        return fn(exp, multiplier, idMap, compute_values, verbose, quadratic)
    if exp.__class__ in native_numeric_types or not exp.is_potentially_variable():
        return _collect_const(exp, multiplier, idMap, compute_values, verbose, quadratic)
    try:
        if exp.is_variable_type():
            fn = _collect_var
        if exp.is_named_expression_type():
            fn = _collect_identity
    except AttributeError:
        pass
    if fn is not None:
        _repn_collectors[exp.__class__] = fn
        return fn(exp, multiplier, idMap, compute_values, verbose, quadratic)
    raise ValueError('Unexpected expression (type %s)' % type(exp).__name__)