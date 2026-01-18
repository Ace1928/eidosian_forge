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
def _get_derived_unit(reg, key):
    try:
        return get_derived_unit(reg, key)
    except KeyError:
        return get_derived_unit(reg, '_'.join(key.split('_')[:-1]))