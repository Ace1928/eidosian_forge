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
def _reg_unique_unit(k, arg_dim, idx):
    if unit_registry is None:
        return
    unique_units[k] = reduce(mul, [1] + [unit_registry[dim] ** v for dim, v in arg_dim[idx].items()])