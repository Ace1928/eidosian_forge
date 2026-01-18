from heapq import nlargest as _nlargest
from collections import namedtuple as _namedtuple
from types import GenericAlias
import re
def _check_types(a, b, *args):
    if a and (not isinstance(a[0], str)):
        raise TypeError('lines to compare must be str, not %s (%r)' % (type(a[0]).__name__, a[0]))
    if b and (not isinstance(b[0], str)):
        raise TypeError('lines to compare must be str, not %s (%r)' % (type(b[0]).__name__, b[0]))
    for arg in args:
        if not isinstance(arg, str):
            raise TypeError('all arguments must be str, not: %r' % (arg,))