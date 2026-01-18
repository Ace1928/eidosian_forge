import functools
import re
import warnings
from abc import ABC, abstractmethod
from Bio import BiopythonDeprecationWarning
from Bio import BiopythonParserWarning
from Bio.Seq import MutableSeq
from Bio.Seq import reverse_complement
from Bio.Seq import Seq
class BetweenPosition(int, Position):
    """Specify the position of a boundary between two coordinates (OBSOLETE?).

    Arguments:
     - position - The default integer position
     - left - The start (left) position of the boundary
     - right - The end (right) position of the boundary

    This allows dealing with a position like 123^456. This
    indicates that the start of the sequence is somewhere between
    123 and 456. It is up to the parser to set the position argument
    to either boundary point (depending on if this is being used as
    a start or end of the feature). For example as a feature end:

    >>> p = BetweenPosition(456, 123, 456)
    >>> p
    BetweenPosition(456, left=123, right=456)
    >>> print(p)
    (123^456)
    >>> int(p)
    456

    Integer equality and comparison use the given position,

    >>> p == 456
    True
    >>> p in [455, 456, 457]
    True
    >>> p > 300
    True

    The old legacy properties of position and extension give the
    starting/lower/left position as an integer, and the distance
    to the ending/higher/right position as an integer. Note that
    the position object will act like either the left or the right
    end-point depending on how it was created:

    >>> p2 = BetweenPosition(123, left=123, right=456)
    >>> int(p) == int(p2)
    False
    >>> p == 456
    True
    >>> p2 == 123
    True

    Note this potentially surprising behavior:

    >>> BetweenPosition(123, left=123, right=456) == ExactPosition(123)
    True
    >>> BetweenPosition(123, left=123, right=456) == BeforePosition(123)
    True
    >>> BetweenPosition(123, left=123, right=456) == AfterPosition(123)
    True

    i.e. For equality (and sorting) the position objects behave like
    integers.

    """

    def __new__(cls, position, left, right):
        """Create a new instance in BetweenPosition object."""
        assert position == left or position == right
        obj = int.__new__(cls, position)
        obj._left = left
        obj._right = right
        return obj

    def __getnewargs__(self):
        """Return the arguments accepted by __new__.

        Necessary to allow pickling and unpickling of class instances.
        """
        return (int(self), self._left, self._right)

    def __repr__(self):
        """Represent the BetweenPosition object as a string for debugging."""
        return '%s(%i, left=%i, right=%i)' % (self.__class__.__name__, int(self), self._left, self._right)

    def __str__(self):
        """Return a representation of the BetweenPosition object (with python counting)."""
        return f'({self._left}^{self._right})'

    def __add__(self, offset):
        """Return a copy of the position object with its location shifted (PRIVATE)."""
        return self.__class__(int(self) + offset, self._left + offset, self._right + offset)

    def _flip(self, length):
        """Return a copy of the location after the parent is reversed (PRIVATE)."""
        return self.__class__(length - int(self), length - self._right, length - self._left)