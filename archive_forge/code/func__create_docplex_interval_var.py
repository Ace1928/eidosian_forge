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
def _create_docplex_interval_var(visitor, interval_var):
    nm = interval_var.name if visitor.symbolic_solver_labels else None
    cpx_interval_var = cp.interval_var(name=nm)
    visitor.var_map[id(interval_var)] = cpx_interval_var
    if interval_var.is_present.fixed and (not interval_var.is_present.value):
        cpx_interval_var.set_absent()
    elif interval_var.optional:
        cpx_interval_var.set_optional()
    else:
        cpx_interval_var.set_present()
    length = interval_var.length
    if length.fixed:
        cpx_interval_var.set_length(length.value)
    if length.lb is not None:
        cpx_interval_var.set_length_min(length.lb)
    if length.ub is not None:
        cpx_interval_var.set_length_max(length.ub)
    start_time = interval_var.start_time
    if start_time.fixed:
        cpx_interval_var.set_start(start_time.value)
    else:
        if start_time.lb is not None:
            cpx_interval_var.set_start_min(start_time.lb)
        if start_time.ub is not None:
            cpx_interval_var.set_start_max(start_time.ub)
    end_time = interval_var.end_time
    if end_time.fixed:
        cpx_interval_var.set_end(end_time.value)
    else:
        if end_time.lb is not None:
            cpx_interval_var.set_end_min(end_time.lb)
        if end_time.ub is not None:
            cpx_interval_var.set_end_max(end_time.ub)
    return cpx_interval_var