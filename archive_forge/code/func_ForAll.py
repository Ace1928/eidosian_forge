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
def ForAll(vs, body, weight=1, qid='', skid='', patterns=[], no_patterns=[]):
    """Create a Z3 forall formula.

    The parameters `weight`, `qid`, `skid`, `patterns` and `no_patterns` are optional annotations.

    >>> f = Function('f', IntSort(), IntSort(), IntSort())
    >>> x = Int('x')
    >>> y = Int('y')
    >>> ForAll([x, y], f(x, y) >= x)
    ForAll([x, y], f(x, y) >= x)
    >>> ForAll([x, y], f(x, y) >= x, patterns=[ f(x, y) ])
    ForAll([x, y], f(x, y) >= x)
    >>> ForAll([x, y], f(x, y) >= x, weight=10)
    ForAll([x, y], f(x, y) >= x)
    """
    return _mk_quantifier(True, vs, body, weight, qid, skid, patterns, no_patterns)