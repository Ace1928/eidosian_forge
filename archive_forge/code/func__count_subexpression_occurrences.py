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
def _count_subexpression_occurrences(self):
    """Categorize named subexpressions based on where they are used.

        This iterates through the `subexpression_order` and categorizes
        each _id based on where it is used (1 constraint, many
        constraints, 1 objective, many objectives, constraints and
        objectives).

        """
    n_subexpressions = [0] * 5
    for info in map(itemgetter(2), map(self.subexpression_cache.__getitem__, self.subexpression_order)):
        if info[2]:
            pass
        elif info[1] is None:
            n_subexpressions[3 if info[0] else 1] += 1
        elif info[0] is None:
            n_subexpressions[4 if info[1] else 2] += 1
        else:
            n_subexpressions[0] += 1
    return n_subexpressions