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
def SetComplement(s):
    """ The complement of set s
    >>> a = Const('a', SetSort(IntSort()))
    >>> SetComplement(a)
    complement(a)
    """
    ctx = s.ctx
    return ArrayRef(Z3_mk_set_complement(ctx.ref(), s.as_ast()), ctx)