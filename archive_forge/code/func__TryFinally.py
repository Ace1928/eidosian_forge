from __future__ import print_function, unicode_literals
import six
import sys
import ast
import os
import tokenize
from six import StringIO
def _TryFinally(self, t):
    if len(t.body) == 1 and isinstance(t.body[0], ast.TryExcept):
        self.dispatch(t.body)
    else:
        self.fill('try')
        self.enter()
        self.dispatch(t.body)
        self.leave()
    self.fill('finally')
    self.enter()
    self.dispatch(t.finalbody)
    self.leave()