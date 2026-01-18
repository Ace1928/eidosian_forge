from __future__ import print_function, unicode_literals
import six
import sys
import ast
import os
import tokenize
from six import StringIO
def _TryExcept(self, t):
    self.fill('try')
    self.enter()
    self.dispatch(t.body)
    self.leave()
    for ex in t.handlers:
        self.dispatch(ex)
    if t.orelse:
        self.fill('else')
        self.enter()
        self.dispatch(t.orelse)
        self.leave()