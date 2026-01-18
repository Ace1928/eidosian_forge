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
def _handle_before_expression_node(visitor, node, time1, time2, delay):
    t1 = (_GENERAL, _time_point_dispatchers[time1[0]](time1[1]))
    t2 = (_GENERAL, _time_point_dispatchers[time2[0]](time2[1]))
    lhs = _handle_sum_node(visitor, None, t1, delay)
    if time1[0] in _non_precedence_types or time2[0] in _non_precedence_types:
        return _handle_inequality_node(visitor, None, lhs, t2)
    return (_BEFORE, _before_dispatchers[time1[0], time2[0]](time1[1], time2[1], delay[1]), (lhs, t2))