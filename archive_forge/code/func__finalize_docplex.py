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
def _finalize_docplex(module, available):
    if not available:
        return
    _deferred_element_getattr_dispatcher['start_time'] = module.start_of
    _deferred_element_getattr_dispatcher['end_time'] = module.end_of
    _deferred_element_getattr_dispatcher['length'] = module.length_of
    _deferred_element_getattr_dispatcher['is_present'] = module.presence_of
    _before_dispatchers[_START_TIME, _START_TIME] = module.start_before_start
    _before_dispatchers[_START_TIME, _END_TIME] = module.start_before_end
    _before_dispatchers[_END_TIME, _START_TIME] = module.end_before_start
    _before_dispatchers[_END_TIME, _END_TIME] = module.end_before_end
    _at_dispatchers[_START_TIME, _START_TIME] = module.start_at_start
    _at_dispatchers[_START_TIME, _END_TIME] = module.start_at_end
    _at_dispatchers[_END_TIME, _START_TIME] = module.end_at_start
    _at_dispatchers[_END_TIME, _END_TIME] = module.end_at_end
    _time_point_dispatchers[_START_TIME] = module.start_of
    _time_point_dispatchers[_END_TIME] = module.end_of