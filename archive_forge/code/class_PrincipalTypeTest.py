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
class PrincipalTypeTest(object):
    """Node test that matches any event with the given principal type."""
    __slots__ = ['principal_type']

    def __init__(self, principal_type):
        self.principal_type = principal_type

    def __call__(self, kind, data, pos, namespaces, variables):
        if kind is START:
            if self.principal_type is ATTRIBUTE:
                return data[1] or None
            else:
                return True

    def __repr__(self):
        return '*'