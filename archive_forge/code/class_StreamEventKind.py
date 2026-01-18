from functools import reduce
import sys
from itertools import chain
import operator
import six
from genshi.compat import stringrepr
from genshi.util import stripentities, striptags
class StreamEventKind(str):
    """A kind of event on a markup stream."""
    __slots__ = []
    _instances = {}

    def __new__(cls, val):
        return cls._instances.setdefault(val, str.__new__(cls, val))