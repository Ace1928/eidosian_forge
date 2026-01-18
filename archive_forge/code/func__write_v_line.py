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
def _write_v_line(self, expr_id, k):
    ostream = self.ostream
    column_order = self.column_order
    info = self.subexpression_cache[expr_id]
    if self.symbolic_solver_labels:
        lbl = '\t#%s' % info[0].name
    else:
        lbl = ''
    self.var_id_to_nl[expr_id] = f'v{self.next_V_line_id}{lbl}'
    linear = dict((item for item in info[1].linear.items() if item[1]))
    ostream.write(f'V{self.next_V_line_id} {len(linear)} {k}{lbl}\n')
    for _id in sorted(linear, key=column_order.__getitem__):
        ostream.write(f'{column_order[_id]} {linear[_id]!r}\n')
    self._write_nl_expression(info[1], True)
    self.next_V_line_id += 1