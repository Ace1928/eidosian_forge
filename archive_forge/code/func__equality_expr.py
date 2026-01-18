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
def _equality_expr(self):
    expr = self._relational_expr()
    while self.cur_token in ('=', '!='):
        op = _operator_map[self.cur_token]
        self.next_token()
        expr = op(expr, self._relational_expr())
    return expr