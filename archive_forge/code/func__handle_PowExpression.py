import itertools
import logging
import math
from io import StringIO
from contextlib import nullcontext
from pyomo.common.collections import OrderedSet
from pyomo.opt import ProblemFormat
from pyomo.opt.base import AbstractProblemWriter, WriterFactory
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.visitor import _ToStringVisitor
import pyomo.core.expr as EXPR
from pyomo.core.base import (
from pyomo.core.base.component import ActiveComponent
import pyomo.core.base.suffix
import pyomo.core.kernel.suffix
from pyomo.core.kernel.block import IBlock
from pyomo.repn.util import valid_expr_ctypes_minlp, valid_active_ctypes_minlp, ftoa
def _handle_PowExpression(visitor, node, values):
    unfixed_count = 0
    for i, arg in enumerate(node.args):
        if type(arg) in native_types:
            pass
        elif arg.is_fixed():
            values[i] = ftoa(value(arg), True)
        else:
            unfixed_count += 1
    if unfixed_count < 2:
        return f'{values[0]} ^ {values[1]}'
    else:
        return f'exp(({values[0]}) * log({values[1]}))'