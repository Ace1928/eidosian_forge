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
class AndOperator(object):
    """The boolean operator `and`."""
    __slots__ = ['lval', 'rval']

    def __init__(self, lval, rval):
        self.lval = lval
        self.rval = rval

    def __call__(self, kind, data, pos, namespaces, variables):
        lval = as_bool(self.lval(kind, data, pos, namespaces, variables))
        if not lval:
            return False
        rval = self.rval(kind, data, pos, namespaces, variables)
        return as_bool(rval)

    def __repr__(self):
        return '%s and %s' % (self.lval, self.rval)