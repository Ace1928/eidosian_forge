from unittest import TestCase
from simplejson.compat import StringIO, long_type, b, binary_type, text_type, PY3
import simplejson as json
class AwesomeInt(object):
    """An awesome reimplementation of integers"""

    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            if isinstance(args[0], int):
                self._int = args[0]

    def __mul__(self, other):
        if hasattr(self, '_int'):
            return self._int * other
        else:
            raise NotImplementedError('To do non-awesome things with this object, please construct it from an integer!')