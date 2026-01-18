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
def _function_call(self):
    name = self.cur_token
    if self.next_token() == '()':
        args = []
    else:
        assert self.cur_token == '('
        self.next_token()
        args = [self._or_expr()]
        while self.cur_token == ',':
            self.next_token()
            args.append(self._or_expr())
        if not self.cur_token == ')':
            raise PathSyntaxError('Expected ")" to close function argument list, but found "%s"' % self.cur_token, self.filename, self.lineno)
    self.next_token()
    cls = _function_map.get(name)
    if not cls:
        raise PathSyntaxError('Unsupported function "%s"' % name, self.filename, self.lineno)
    return cls(*args)