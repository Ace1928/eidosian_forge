import sys
from typing import Dict
class StaticTuple(tuple):
    """A static type, similar to a tuple of strings."""
    __slots__ = ()

    def __new__(cls, *args):
        if not args and _empty_tuple is not None:
            return _empty_tuple
        return tuple.__new__(cls, args)

    def __init__(self, *args):
        """Create a new 'StaticTuple'"""
        num_keys = len(args)
        if num_keys < 0 or num_keys > 255:
            raise TypeError('StaticTuple(...) takes from 0 to 255 items')
        for bit in args:
            if type(bit) not in _valid_types:
                raise TypeError('StaticTuple can only point to StaticTuple, str, unicode, int, float, bool, or None not %s' % (type(bit),))
        tuple.__init__(self)

    def __repr__(self):
        return '{}{}'.format(self.__class__.__name__, tuple.__repr__(self))

    def __reduce__(self):
        return (StaticTuple, tuple(self))

    def __add__(self, other):
        """Concatenate self with other"""
        return StaticTuple.from_sequence(tuple.__add__(self, other))

    def as_tuple(self):
        return tuple(self)

    def intern(self):
        return _interned_tuples.setdefault(self, self)

    @staticmethod
    def from_sequence(seq):
        """Convert a sequence object into a StaticTuple instance."""
        if isinstance(seq, StaticTuple):
            return seq
        return StaticTuple(*seq)