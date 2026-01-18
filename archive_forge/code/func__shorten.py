import itertools
import types
import unittest
from cupy.testing import _bundle
from cupy.testing import _pytest_impl
def _shorten(s, maxlen):
    ellipsis = '...'
    if len(s) <= maxlen:
        return s
    n1 = (maxlen - len(ellipsis)) // 2
    n2 = maxlen - len(ellipsis) - n1
    s = s[:n1] + ellipsis + s[-n2:]
    assert len(s) == maxlen
    return s