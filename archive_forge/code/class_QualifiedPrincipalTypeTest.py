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
class QualifiedPrincipalTypeTest(object):
    """Node test that matches any event with the given principal type in a
    specific namespace."""
    __slots__ = ['principal_type', 'prefix']

    def __init__(self, principal_type, prefix):
        self.principal_type = principal_type
        self.prefix = prefix

    def __call__(self, kind, data, pos, namespaces, variables):
        namespace = Namespace(namespaces.get(self.prefix))
        if kind is START:
            if self.principal_type is ATTRIBUTE and data[1]:
                return Attrs([(name, value) for name, value in data[1] if name in namespace]) or None
            else:
                return data[0] in namespace

    def __repr__(self):
        return '%s:*' % self.prefix