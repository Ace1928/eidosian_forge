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
def _location_step(self):
    if self.cur_token == '@':
        axis = ATTRIBUTE
        self.next_token()
    elif self.cur_token == '.':
        axis = SELF
    elif self.cur_token == '..':
        raise PathSyntaxError('Unsupported axis "parent"', self.filename, self.lineno)
    elif self.peek_token() == '::':
        axis = Axis.forname(self.cur_token)
        if axis is None:
            raise PathSyntaxError('Unsupport axis "%s"' % axis, self.filename, self.lineno)
        self.next_token()
        self.next_token()
    else:
        axis = None
    nodetest = self._node_test(axis or CHILD)
    predicates = []
    while self.cur_token == '[':
        predicates.append(self._predicate())
    return (axis, nodetest, predicates)