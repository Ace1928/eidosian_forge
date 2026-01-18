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
def _collect_prod(exp, multiplier, idMap, compute_values, verbose, quadratic):
    if exp._args_[0].__class__ in native_numeric_types:
        if exp._args_[0] == 0:
            return Results()
        return _collect_standard_repn(exp._args_[1], multiplier * exp._args_[0], idMap, compute_values, verbose, quadratic)
    if exp._args_[1].__class__ in native_numeric_types:
        if exp._args_[1] == 0:
            return Results()
        return _collect_standard_repn(exp._args_[0], multiplier * exp._args_[1], idMap, compute_values, verbose, quadratic)
    elif not exp._args_[0].is_potentially_variable():
        if compute_values:
            val = value(exp._args_[0])
            if val == 0:
                return Results()
            return _collect_standard_repn(exp._args_[1], multiplier * val, idMap, compute_values, verbose, quadratic)
        else:
            return _collect_standard_repn(exp._args_[1], multiplier * exp._args_[0], idMap, compute_values, verbose, quadratic)
    elif not exp._args_[1].is_potentially_variable():
        if compute_values:
            val = value(exp._args_[1])
            if val == 0:
                return Results()
            return _collect_standard_repn(exp._args_[0], multiplier * val, idMap, compute_values, verbose, quadratic)
        else:
            return _collect_standard_repn(exp._args_[0], multiplier * exp._args_[1], idMap, compute_values, verbose, quadratic)
    lhs = _collect_standard_repn(exp._args_[0], 1, idMap, compute_values, verbose, quadratic)
    lhs_nonl_None = lhs.nonl.__class__ in native_numeric_types and (not lhs.nonl)
    if lhs_nonl_None and len(lhs.linear) == 0 and (not quadratic or len(lhs.quadratic) == 0):
        if lhs.constant.__class__ in native_numeric_types and lhs.constant == 0:
            return Results()
        if compute_values:
            val = value(lhs.constant)
            if val == 0:
                return Results()
            return _collect_standard_repn(exp._args_[1], multiplier * val, idMap, compute_values, verbose, quadratic)
        else:
            return _collect_standard_repn(exp._args_[1], multiplier * lhs.constant, idMap, compute_values, verbose, quadratic)
    rhs = _collect_standard_repn(exp._args_[1], 1, idMap, compute_values, verbose, quadratic)
    rhs_nonl_None = rhs.nonl.__class__ in native_numeric_types and (not rhs.nonl)
    if rhs_nonl_None and len(rhs.linear) == 0 and (not quadratic or len(rhs.quadratic) == 0) and (rhs.constant.__class__ in native_numeric_types) and (rhs.constant == 0):
        return Results()
    if not lhs_nonl_None or not rhs_nonl_None:
        return Results(nonl=multiplier * exp)
    if max(1 if lhs.linear else 0, 2 if quadratic and lhs.quadratic else 0) + max(1 if rhs.linear else 0, 2 if quadratic and rhs.quadratic else 0) > (2 if quadratic else 1):
        return Results(nonl=multiplier * exp)
    ans = Results()
    ans.constant = multiplier * lhs.constant * rhs.constant
    if not (lhs.constant.__class__ in native_numeric_types and lhs.constant == 0):
        for key, coef in rhs.linear.items():
            ans.linear[key] = multiplier * coef * lhs.constant
    if not (rhs.constant.__class__ in native_numeric_types and rhs.constant == 0):
        for key, coef in lhs.linear.items():
            if key in ans.linear:
                ans.linear[key] += multiplier * coef * rhs.constant
            else:
                ans.linear[key] = multiplier * coef * rhs.constant
    if quadratic:
        if not (lhs.constant.__class__ in native_numeric_types and lhs.constant == 0):
            for key, coef in rhs.quadratic.items():
                ans.quadratic[key] = multiplier * coef * lhs.constant
        if not (rhs.constant.__class__ in native_numeric_types and rhs.constant == 0):
            for key, coef in lhs.quadratic.items():
                if key in ans.quadratic:
                    ans.quadratic[key] += multiplier * coef * rhs.constant
                else:
                    ans.quadratic[key] = multiplier * coef * rhs.constant
        for lkey, lcoef in lhs.linear.items():
            for rkey, rcoef in rhs.linear.items():
                ndx = (lkey, rkey) if lkey <= rkey else (rkey, lkey)
                if ndx in ans.quadratic:
                    ans.quadratic[ndx] += multiplier * lcoef * rcoef
                else:
                    ans.quadratic[ndx] = multiplier * lcoef * rcoef
        el_linear = multiplier * sum((coef * idMap[key] for key, coef in lhs.linear.items() if coef.__class__ not in native_numeric_types or coef))
        er_linear = multiplier * sum((coef * idMap[key] for key, coef in rhs.linear.items() if coef.__class__ not in native_numeric_types or coef))
        el_quadratic = multiplier * sum((coef * idMap[key[0]] * idMap[key[1]] for key, coef in lhs.quadratic.items() if coef.__class__ not in native_numeric_types or coef))
        er_quadratic = multiplier * sum((coef * idMap[key[0]] * idMap[key[1]] for key, coef in rhs.quadratic.items() if coef.__class__ not in native_numeric_types or coef))
        if (el_linear.__class__ not in native_numeric_types or el_linear) and (er_quadratic.__class__ not in native_numeric_types or er_quadratic):
            ans.nonl += el_linear * er_quadratic
        if (er_linear.__class__ not in native_numeric_types or er_linear) and (el_quadratic.__class__ not in native_numeric_types or el_quadratic):
            ans.nonl += er_linear * el_quadratic
    return ans