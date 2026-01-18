import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
class SortAnythingKey(object):

    def __init__(self, obj):
        self.obj = obj

    def _python_lt(self, other_obj):
        try:
            return self.obj < other_obj
        except TypeError:
            return NotImplemented

    def __lt__(self, other):
        assert isinstance(other, SortAnythingKey)
        result = self._python_lt(other.obj)
        if result is not NotImplemented:
            return result
        if self._python_lt(0) is not NotImplemented:
            return True
        if other._python_lt(0) is not NotImplemented:
            return False
        if self.obj == other.obj:
            return False
        return (self.obj.__class__.__name__, id(self.obj)) < (other.obj.__class__.__name__, id(other.obj))