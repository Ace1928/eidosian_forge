from __future__ import absolute_import, print_function, division
import logging
from petl.compat import callable
def _hasmethod(o, n):
    return hasattr(o, n) and callable(getattr(o, n))