from __future__ import division
import sys
import unicodedata
from functools import reduce
@classmethod
def _fmt_float(cls, x, **kw):
    """Float formatting class-method.

        - x parameter is ignored. Instead kw-argument f being x float-converted
          will be used.

        - precision will be taken from `n` kw-argument.
        """
    n = kw.get('n')
    return '%.*f' % (n, cls._to_float(x))