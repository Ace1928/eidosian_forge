import ctypes
import logging
import os
from collections import deque, defaultdict, namedtuple
from contextlib import nullcontext
from itertools import filterfalse, product
from math import log10 as _log10
from operator import itemgetter, attrgetter, setitem
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.config import (
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.errors import DeveloperError, InfeasibleConstraintException, MouseTrap
from pyomo.common.gc_manager import PauseGC
from pyomo.common.numeric_types import (
from pyomo.common.timing import TicTocTimer
from pyomo.core.expr import (
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, _EvaluationVisitor
from pyomo.core.base import (
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.constraint import _ConstraintData
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.objective import (
from pyomo.core.base.suffix import SuffixFinder
from pyomo.core.base.var import _VarData
import pyomo.core.kernel as kernel
from pyomo.core.pyomoobject import PyomoObject
from pyomo.opt import WriterFactory
from pyomo.repn.util import (
from pyomo.repn.plugins.ampl.ampl_ import set_pyomo_amplfunc_env
from pyomo.core.base import Set, RangeSet
from pyomo.network import Port
class _SuffixData(object):

    def __init__(self, name):
        self.name = name
        self.obj = {}
        self.con = {}
        self.var = {}
        self.prob = {}
        self.datatype = set()
        self.values = ComponentMap()

    def update(self, suffix):
        self.datatype.add(suffix.datatype)
        self.values.update(suffix)

    def store(self, obj, val):
        self.values[obj] = val

    def compile(self, column_order, row_order, obj_order, model_id):
        missing_component_data = ComponentSet()
        unknown_data = ComponentSet()
        queue = [self.values.items()]
        while queue:
            for obj, val in queue.pop(0):
                if val.__class__ not in int_float:
                    if isinstance(val, dict):
                        queue.append(val.items())
                        continue
                    val = float(val)
                _id = id(obj)
                if _id in column_order:
                    self.var[column_order[_id]] = val
                elif _id in row_order:
                    self.con[row_order[_id]] = val
                elif _id in obj_order:
                    self.obj[obj_order[_id]] = val
                elif _id == model_id:
                    self.prob[0] = val
                elif isinstance(obj, (_VarData, _ConstraintData, _ObjectiveData)):
                    missing_component_data.add(obj)
                elif isinstance(obj, (Var, Constraint, Objective)):
                    queue.append(product(filterfalse(self.values.__contains__, obj.values()), (val,)))
                else:
                    unknown_data.add(obj)
        if missing_component_data:
            logger.warning(f"model contains export suffix '{self.name}' that contains {len(missing_component_data)} component keys that are not exported as part of the NL file.  Skipping.")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Skipped component keys:\n\t' + '\n\t'.join(sorted(map(str, missing_component_data))))
        if unknown_data:
            logger.warning(f"model contains export suffix '{self.name}' that contains {len(unknown_data)} keys that are not Var, Constraint, Objective, or the model.  Skipping.")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Skipped component keys:\n\t' + '\n\t'.join(sorted(map(str, unknown_data))))