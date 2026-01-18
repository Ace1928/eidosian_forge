import functools
import re
import warnings
@functools.total_ordering
class NumericIdentifier(object):
    __slots__ = ['value']

    def __init__(self, value):
        self.value = int(value)

    def __repr__(self):
        return 'NumericIdentifier(%r)' % self.value

    def __eq__(self, other):
        if isinstance(other, NumericIdentifier):
            return self.value == other.value
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, MaxIdentifier):
            return True
        elif isinstance(other, AlphaIdentifier):
            return True
        elif isinstance(other, NumericIdentifier):
            return self.value < other.value
        else:
            return NotImplemented