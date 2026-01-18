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
class NodeTest(object):
    """Node test that matches any node."""
    __slots__ = []

    def __call__(self, kind, data, pos, namespaces, variables):
        if kind is START:
            return True
        return (kind, data, pos)

    def __repr__(self):
        return 'node()'