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
class NormalizeSpaceFunction(Function):
    """The `normalize-space` function, which removes leading and trailing
    whitespace in the given string, and replaces multiple adjacent whitespace
    characters inside the string with a single space.
    """
    __slots__ = ['expr']
    _normalize = re.compile('\\s{2,}').sub

    def __init__(self, expr):
        self.expr = expr

    def __call__(self, kind, data, pos, namespaces, variables):
        string = self.expr(kind, data, pos, namespaces, variables)
        return self._normalize(' ', as_string(string).strip())

    def __repr__(self):
        return 'normalize-space(%s)' % repr(self.expr)