import functools
import re
import warnings
from abc import ABC, abstractmethod
from Bio import BiopythonDeprecationWarning
from Bio import BiopythonParserWarning
from Bio.Seq import MutableSeq
from Bio.Seq import reverse_complement
from Bio.Seq import Seq
class WithinPosition(int, Position):
    """Specify the position of a boundary within some coordinates.

    Arguments:
    - position - The default integer position
    - left - The start (left) position of the boundary
    - right - The end (right) position of the boundary

    This allows dealing with a location like ((11.14)..100). This
    indicates that the start of the sequence is somewhere between 11
    and 14. Since this is a start coordinate, it should act like
    it is at position 11 (or in Python counting, 10).

    >>> p = WithinPosition(10, 10, 13)
    >>> p
    WithinPosition(10, left=10, right=13)
    >>> print(p)
    (10.13)
    >>> int(p)
    10

    Basic integer comparisons and operations should work as though
    this were a plain integer:

    >>> p == 10
    True
    >>> p in [9, 10, 11]
    True
    >>> p < 11
    True
    >>> p + 10
    WithinPosition(20, left=20, right=23)

    >>> isinstance(p, WithinPosition)
    True
    >>> isinstance(p, Position)
    True
    >>> isinstance(p, int)
    True

    Note this also applies for comparison to other position objects,
    where again the integer behavior is used:

    >>> p == 10
    True
    >>> p == ExactPosition(10)
    True
    >>> p == BeforePosition(10)
    True
    >>> p == AfterPosition(10)
    True

    If this were an end point, you would want the position to be 13
    (the right/larger value, not the left/smaller value as above):

    >>> p2 = WithinPosition(13, 10, 13)
    >>> p2
    WithinPosition(13, left=10, right=13)
    >>> print(p2)
    (10.13)
    >>> int(p2)
    13
    >>> p2 == 13
    True
    >>> p2 == ExactPosition(13)
    True

    """

    def __new__(cls, position, left, right):
        """Create a WithinPosition object."""
        if not (position == left or position == right):
            raise RuntimeError('WithinPosition: %r should match left %r or right %r' % (position, left, right))
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
        """Represent the WithinPosition object as a string for debugging."""
        return '%s(%i, left=%i, right=%i)' % (self.__class__.__name__, int(self), self._left, self._right)

    def __str__(self):
        """Return a representation of the WithinPosition object (with python counting)."""
        return f'({self._left}.{self._right})'

    def __add__(self, offset):
        """Return a copy of the position object with its location shifted."""
        return self.__class__(int(self) + offset, self._left + offset, self._right + offset)

    def _flip(self, length):
        """Return a copy of the location after the parent is reversed (PRIVATE)."""
        return self.__class__(length - int(self), length - self._right, length - self._left)