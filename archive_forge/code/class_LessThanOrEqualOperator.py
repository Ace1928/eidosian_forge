from collections import deque
from functools import reduce
from math import ceil, floor
import operator
import re
from itertools import chain
import six
from genshi.compat import IS_PYTHON2
from genshi.core import Stream, Attrs, Namespace, QName
from genshi.core import START, END, TEXT, START_NS, END_NS, COMMENT, PI, \
class LessThanOrEqualOperator(object):
    """The relational operator `<=` (less than or equal)."""
    __slots__ = ['lval', 'rval']

    def __init__(self, lval, rval):
        self.lval = lval
        self.rval = rval

    def __call__(self, kind, data, pos, namespaces, variables):
        lval = self.lval(kind, data, pos, namespaces, variables)
        rval = self.rval(kind, data, pos, namespaces, variables)
        return as_float(lval) <= as_float(rval)

    def __repr__(self):
        return '%s<=%s' % (self.lval, self.rval)