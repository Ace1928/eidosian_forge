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
def _or_expr(self):
    expr = self._and_expr()
    while self.cur_token == 'or':
        self.next_token()
        expr = OrOperator(expr, self._and_expr())
    return expr