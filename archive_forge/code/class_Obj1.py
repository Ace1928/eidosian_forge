import re
import pytest  # NOQA
from .roundtrip import dedent
class Obj1(dict):

    def __init__(self, suffix):
        self._suffix = suffix
        self._node = None

    def add_node(self, n):
        self._node = n

    def __repr__(self):
        return 'Obj1(%s->%s)' % (self._suffix, self.items())

    def dump(self):
        return repr(self._node)