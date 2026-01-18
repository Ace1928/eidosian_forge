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
def PrefixOf(a, b):
    """Check if 'a' is a prefix of 'b'
    >>> s1 = PrefixOf("ab", "abc")
    >>> simplify(s1)
    True
    >>> s2 = PrefixOf("bc", "abc")
    >>> simplify(s2)
    False
    """
    ctx = _get_ctx2(a, b)
    a = _coerce_seq(a, ctx)
    b = _coerce_seq(b, ctx)
    return BoolRef(Z3_mk_seq_prefix(a.ctx_ref(), a.as_ast(), b.as_ast()), a.ctx)