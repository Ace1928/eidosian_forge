from pyomo.common.dependencies import attempt_import
import itertools
import logging
from operator import attrgetter
from pyomo.common import DeveloperError
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.collections import ComponentMap
from pyomo.common.fileutils import Executable
from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.interval_var import (
from pyomo.contrib.cp.scheduling_expr.precedence_expressions import (
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.core.base import (
from pyomo.core.base.boolean_var import (
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.param import IndexedParam, ScalarParam, _ParamData
from pyomo.core.base.var import ScalarVar, _GeneralVarData, IndexedVar
import pyomo.core.expr as EXPR
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, identify_variables
from pyomo.core.base import Set, RangeSet
from pyomo.core.base.set import SetProduct
from pyomo.opt import WriterFactory, SolverFactory, TerminationCondition, SolverResults
from pyomo.network import Port
def _handle_getitem(visitor, node, *data):
    arg_domain = []
    arg_scale = []
    expr = 0
    mult = 1
    for i, arg in enumerate(data[1:]):
        if arg[1].__class__ in EXPR.native_types:
            arg_set = Set(initialize=[arg[1]])
            arg_set.construct()
            arg_domain.append(arg_set)
            arg_scale.append(None)
        elif node.arg(i + 1).is_expression_type():
            arg_expr = node.arg(i + 1)
            var_list = list(identify_variables(arg_expr, include_fixed=False))
            var_domain = [list(_check_var_domain(visitor, node, v)) for v in var_list]
            arg_vals = set()
            for var_vals in itertools.product(*var_domain):
                for v, val in zip(var_list, var_vals):
                    v.set_value(val)
                arg_vals.add(arg_expr())
            arg_set = Set(initialize=sorted(arg_vals))
            arg_set.construct()
            interval = arg_set.get_interval()
            if not interval[2]:
                raise ValueError("Variable indirection '%s' contains argument expression '%s' that does not evaluate to a simple discrete set" % (node, arg_expr))
            arg_domain.append(arg_set)
            arg_scale.append(interval)
        else:
            var = node.arg(i + 1)
            arg_domain.append(_check_var_domain(visitor, node, var))
            arg_scale.append(arg_domain[-1].get_interval())
        if arg_scale[-1] is not None:
            _min, _max, _step = arg_scale[-1]
            if _step is None:
                raise ValueError("Variable indirection '%s' is over a discrete domain without a constant step size. This is not supported." % node)
            expr += mult * (arg[1] - _min) // _step
            mult *= len(arg_domain[-1])
    elements = []
    for idx in SetProduct(*arg_domain):
        try:
            idx = idx if len(idx) > 1 else idx[0]
            elements.append(data[0][1][idx])
        except KeyError:
            raise ValueError("Variable indirection '%s' permits an index '%s' that is not a valid key. In CP Optimizer, this is a structural infeasibility." % (node, idx))
    try:
        return (_ELEMENT_CONSTRAINT, cp.element(elements, expr))
    except AssertionError:
        return (_DEFERRED_ELEMENT_CONSTRAINT, (elements, expr))