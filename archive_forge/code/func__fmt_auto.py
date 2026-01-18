from __future__ import division
import sys
import unicodedata
from functools import reduce
@classmethod
def _fmt_auto(cls, x, **kw):
    """auto formatting class-method."""
    f = cls._to_float(x)
    if abs(f) > 100000000.0:
        fn = cls._fmt_exp
    elif f != f:
        fn = cls._fmt_text
    elif f - round(f) == 0:
        fn = cls._fmt_bool if isinstance(x, bool) else cls._fmt_int
    else:
        fn = cls._fmt_float
    return fn(x, **kw)