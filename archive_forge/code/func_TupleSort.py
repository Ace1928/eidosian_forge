from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
def TupleSort(name, sorts, ctx=None):
    """Create a named tuple sort base on a set of underlying sorts
    Example:
        >>> pair, mk_pair, (first, second) = TupleSort("pair", [IntSort(), StringSort()])
    """
    tuple = Datatype(name, ctx)
    projects = [('project%d' % i, sorts[i]) for i in range(len(sorts))]
    tuple.declare(name, *projects)
    tuple = tuple.create()
    return (tuple, tuple.constructor(0), [tuple.accessor(0, i) for i in range(len(sorts))])