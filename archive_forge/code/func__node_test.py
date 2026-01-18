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
def _node_test(self, axis=None):
    test = prefix = None
    next_token = self.peek_token()
    if next_token in ('(', '()'):
        test = self._node_type()
    elif next_token == ':':
        prefix = self.cur_token
        self.next_token()
        localname = self.next_token()
        if localname == '*':
            test = QualifiedPrincipalTypeTest(axis, prefix)
        else:
            test = QualifiedNameTest(axis, prefix, localname)
    elif self.cur_token == '*':
        test = PrincipalTypeTest(axis)
    elif self.cur_token == '.':
        test = NodeTest()
    else:
        test = LocalNameTest(axis, self.cur_token)
    if not self.at_end:
        self.next_token()
    return test