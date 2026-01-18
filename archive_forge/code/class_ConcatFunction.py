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
class ConcatFunction(Function):
    """The `concat` function, which concatenates (joins) the variable number of
    strings it gets as arguments.
    """
    __slots__ = ['exprs']

    def __init__(self, *exprs):
        self.exprs = exprs

    def __call__(self, kind, data, pos, namespaces, variables):
        strings = []
        for item in [expr(kind, data, pos, namespaces, variables) for expr in self.exprs]:
            strings.append(as_string(item))
        return ''.join(strings)

    def __repr__(self):
        return 'concat(%s)' % ', '.join([repr(expr) for expr in self.exprs])