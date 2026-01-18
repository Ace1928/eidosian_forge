import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
class PushbackAdapter(object):

    def __init__(self, it):
        self._it = it
        self._pushed = []

    def __iter__(self):
        return self

    def push_back(self, obj):
        self._pushed.append(obj)

    def next(self):
        if self._pushed:
            return self._pushed.pop()
        else:
            return six.advance_iterator(self._it)
    __next__ = next

    def peek(self):
        try:
            obj = six.advance_iterator(self)
        except StopIteration:
            raise ValueError('no more data')
        self.push_back(obj)
        return obj

    def has_more(self):
        try:
            self.peek()
        except ValueError:
            return False
        else:
            return True