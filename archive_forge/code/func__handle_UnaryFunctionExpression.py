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
def _handle_UnaryFunctionExpression(visitor, node, values):
    if node.name == 'sqrt':
        return f'(({values[0]}) ^ 0.5)'
    elif node.name == 'log10':
        return f'({_log10_e} * log({values[0]}))'
    elif node.name not in _allowableUnaryFunctions:
        raise RuntimeError('The BARON .BAR format does not support the unary function "%s".' % (node.name,))
    return node._to_string(values, visitor.verbose, visitor.smap)