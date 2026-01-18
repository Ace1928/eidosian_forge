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
class CeilingFunction(Function):
    """The `ceiling` function, which returns the nearest lower integer number
    for the given number.
    """
    __slots__ = ['number']

    def __init__(self, number):
        self.number = number

    def __call__(self, kind, data, pos, namespaces, variables):
        number = self.number(kind, data, pos, namespaces, variables)
        return ceil(as_float(number))

    def __repr__(self):
        return 'ceiling(%r)' % self.number