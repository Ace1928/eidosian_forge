import math
from collections.abc import Sequence
from pyomo.common.numeric_types import check_if_numeric_type
class NonNumericRange(object):
    """A range-like object for representing a single non-numeric value

    The class name is a bit of a misnomer, as this object does not
    represent a range but rather a single value.  However, as it
    duplicates the Range API (as used by :py:class:`NumericRange`), it
    is called a "Range".

    """
    __slots__ = ('value',)

    def __init__(self, val):
        self.value = val

    def __str__(self):
        return '{%s}' % (self.value,)
    __repr__ = __str__

    def __eq__(self, other):
        return isinstance(other, NonNumericRange) and other.value == self.value

    def __ne__(self, other):
        return not self.__eq__(other)

    def __contains__(self, value):
        return value == self.value

    def __getstate__(self):
        """
        Retrieve the state of this object as a dictionary.

        This method must be defined because this class uses slots.
        """
        state = {}
        for i in NonNumericRange.__slots__:
            state[i] = getattr(self, i)
        return state

    def __setstate__(self, state):
        """
        Set the state of this object using values from a state dictionary.

        This method must be defined because this class uses slots.
        """
        for key, val in state.items():
            object.__setattr__(self, key, val)

    def isdiscrete(self):
        return True

    def isfinite(self):
        return True

    def isdisjoint(self, other):
        return self.value not in other

    def issubset(self, other):
        return self.value in other

    def range_difference(self, other_ranges):
        for r in other_ranges:
            if self.value in r:
                return []
        return [self]

    def range_intersection(self, other_ranges):
        for r in other_ranges:
            if self.value in r:
                return [self]
        return []