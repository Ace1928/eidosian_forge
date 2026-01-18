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
def _create_docplex_var(pyomo_var, name=None):
    if pyomo_var.is_binary():
        return cp.binary_var(name=name)
    elif pyomo_var.is_integer():
        return cp.integer_var(min=pyomo_var.bounds[0], max=pyomo_var.bounds[1], name=name)
    elif pyomo_var.domain.isdiscrete():
        if pyomo_var.domain.isfinite():
            return cp.integer_var(domain=[d for d in pyomo_var.domain], name=name)
        else:
            raise ValueError("The LogicalToDoCplex writer does not support infinite discrete domains. Cannot write Var '%s' with domain '%s'" % (pyomo_var.name, pyomo_var.domain))
    else:
        raise ValueError("The LogicalToDoCplex writer can only support integer- or Boolean-valued variables. Cannot write Var '%s' with domain '%s'" % (pyomo_var.name, pyomo_var.domain))