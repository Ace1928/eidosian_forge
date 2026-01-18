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
class VariableReference(Literal):
    """A variable reference node."""
    __slots__ = ['name']

    def __init__(self, name):
        self.name = name

    def __call__(self, kind, data, pos, namespaces, variables):
        return variables.get(self.name)

    def __repr__(self):
        return str(self.name)