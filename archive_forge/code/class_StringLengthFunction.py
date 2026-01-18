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
class StringLengthFunction(Function):
    """The `string-length` function that returns the length of the given
    string.
    """
    __slots__ = ['expr']

    def __init__(self, expr):
        self.expr = expr

    def __call__(self, kind, data, pos, namespaces, variables):
        string = self.expr(kind, data, pos, namespaces, variables)
        return len(as_string(string))

    def __repr__(self):
        return 'string-length(%r)' % self.expr