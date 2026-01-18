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
def SuffixOf(a, b):
    """Check if 'a' is a suffix of 'b'
    >>> s1 = SuffixOf("ab", "abc")
    >>> simplify(s1)
    False
    >>> s2 = SuffixOf("bc", "abc")
    >>> simplify(s2)
    True
    """
    ctx = _get_ctx2(a, b)
    a = _coerce_seq(a, ctx)
    b = _coerce_seq(b, ctx)
    return BoolRef(Z3_mk_seq_suffix(a.ctx_ref(), a.as_ast(), b.as_ast()), a.ctx)