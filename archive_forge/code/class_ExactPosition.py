import functools
import re
import warnings
from abc import ABC, abstractmethod
from Bio import BiopythonDeprecationWarning
from Bio import BiopythonParserWarning
from Bio.Seq import MutableSeq
from Bio.Seq import reverse_complement
from Bio.Seq import Seq
class ExactPosition(int, Position):
    """Specify the specific position of a boundary.

    Arguments:
     - position - The position of the boundary.
     - extension - An optional argument which must be zero since we don't
       have an extension. The argument is provided so that the same number
       of arguments can be passed to all position types.

    In this case, there is no fuzziness associated with the position.

    >>> p = ExactPosition(5)
    >>> p
    ExactPosition(5)
    >>> print(p)
    5

    >>> isinstance(p, Position)
    True
    >>> isinstance(p, int)
    True

    Integer comparisons and operations should work as expected:

    >>> p == 5
    True
    >>> p < 6
    True
    >>> p <= 5
    True
    >>> p + 10
    ExactPosition(15)

    """

    def __new__(cls, position, extension=0):
        """Create an ExactPosition object."""
        if extension != 0:
            raise AttributeError(f'Non-zero extension {extension} for exact position.')
        return int.__new__(cls, position)

    def __str__(self):
        """Return a representation of the ExactPosition object (with python counting)."""
        return str(int(self))

    def __repr__(self):
        """Represent the ExactPosition object as a string for debugging."""
        return '%s(%i)' % (self.__class__.__name__, int(self))

    def __add__(self, offset):
        """Return a copy of the position object with its location shifted (PRIVATE)."""
        return self.__class__(int(self) + offset)

    def _flip(self, length):
        """Return a copy of the location after the parent is reversed (PRIVATE)."""
        return self.__class__(length - int(self))