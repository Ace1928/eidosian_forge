from collections import OrderedDict
from functools import reduce, partial
from itertools import chain
from operator import attrgetter, mul
import math
import warnings
from ..units import (
from ..util.pyutil import deprecated
from ..util._expr import Expr, Symbol
from .rates import RateExpr, MassAction
def _get_arg_dim(expr, rxn):
    if unit_registry is None:
        return None
    else:
        return expr.args_dimensionality(reaction=rxn)