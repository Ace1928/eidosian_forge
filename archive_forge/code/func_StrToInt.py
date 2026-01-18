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
def StrToInt(s):
    """Convert string expression to integer
    >>> a = StrToInt("1")
    >>> simplify(1 == a)
    True
    >>> b = StrToInt("2")
    >>> simplify(1 == b)
    False
    >>> c = StrToInt(IntToStr(2))
    >>> simplify(1 == c)
    False
    """
    s = _coerce_seq(s)
    return ArithRef(Z3_mk_str_to_int(s.ctx_ref(), s.as_ast()), s.ctx)