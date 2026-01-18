from __future__ import division
import sys
import unicodedata
from functools import reduce
@classmethod
def _fmt_int(cls, x, **kw):
    """Integer formatting class-method.
        """
    if type(x) == int:
        return str(x)
    else:
        return str(int(round(cls._to_float(x))))