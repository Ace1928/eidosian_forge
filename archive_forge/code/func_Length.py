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
def Length(s):
    """Obtain the length of a sequence 's'
    >>> l = Length(StringVal("abc"))
    >>> simplify(l)
    3
    """
    s = _coerce_seq(s)
    return ArithRef(Z3_mk_seq_length(s.ctx_ref(), s.as_ast()), s.ctx)