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
def Re(s, ctx=None):
    """The regular expression that accepts sequence 's'
    >>> s1 = Re("ab")
    >>> s2 = Re(StringVal("ab"))
    >>> s3 = Re(Unit(BoolVal(True)))
    """
    s = _coerce_seq(s, ctx)
    return ReRef(Z3_mk_seq_to_re(s.ctx_ref(), s.as_ast()), s.ctx)