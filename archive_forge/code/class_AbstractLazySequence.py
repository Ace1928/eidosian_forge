import bisect
from collections import Counter, defaultdict, deque
from functools import total_ordering
from itertools import chain, islice
from nltk.internals import raise_unorderable_types, slice_bounds
@total_ordering
class AbstractLazySequence:
    """
    An abstract base class for read-only sequences whose values are
    computed as needed.  Lazy sequences act like tuples -- they can be
    indexed, sliced, and iterated over; but they may not be modified.

    The most common application of lazy sequences in NLTK is for
    corpus view objects, which provide access to the contents of a
    corpus without loading the entire corpus into memory, by loading
    pieces of the corpus from disk as needed.

    The result of modifying a mutable element of a lazy sequence is
    undefined.  In particular, the modifications made to the element
    may or may not persist, depending on whether and when the lazy
    sequence caches that element's value or reconstructs it from
    scratch.

    Subclasses are required to define two methods: ``__len__()``
    and ``iterate_from()``.
    """

    def __len__(self):
        """
        Return the number of tokens in the corpus file underlying this
        corpus view.
        """
        raise NotImplementedError('should be implemented by subclass')

    def iterate_from(self, start):
        """
        Return an iterator that generates the tokens in the corpus
        file underlying this corpus view, starting at the token number
        ``start``.  If ``start>=len(self)``, then this iterator will
        generate no tokens.
        """
        raise NotImplementedError('should be implemented by subclass')

    def __getitem__(self, i):
        """
        Return the *i* th token in the corpus file underlying this
        corpus view.  Negative indices and spans are both supported.
        """
        if isinstance(i, slice):
            start, stop = slice_bounds(self, i)
            return LazySubsequence(self, start, stop)
        else:
            if i < 0:
                i += len(self)
            if i < 0:
                raise IndexError('index out of range')
            try:
                return next(self.iterate_from(i))
            except StopIteration as e:
                raise IndexError('index out of range') from e

    def __iter__(self):
        """Return an iterator that generates the tokens in the corpus
        file underlying this corpus view."""
        return self.iterate_from(0)

    def count(self, value):
        """Return the number of times this list contains ``value``."""
        return sum((1 for elt in self if elt == value))

    def index(self, value, start=None, stop=None):
        """Return the index of the first occurrence of ``value`` in this
        list that is greater than or equal to ``start`` and less than
        ``stop``.  Negative start and stop values are treated like negative
        slice bounds -- i.e., they count from the end of the list."""
        start, stop = slice_bounds(self, slice(start, stop))
        for i, elt in enumerate(islice(self, start, stop)):
            if elt == value:
                return i + start
        raise ValueError('index(x): x not in list')

    def __contains__(self, value):
        """Return true if this list contains ``value``."""
        return bool(self.count(value))

    def __add__(self, other):
        """Return a list concatenating self with other."""
        return LazyConcatenation([self, other])

    def __radd__(self, other):
        """Return a list concatenating other with self."""
        return LazyConcatenation([other, self])

    def __mul__(self, count):
        """Return a list concatenating self with itself ``count`` times."""
        return LazyConcatenation([self] * count)

    def __rmul__(self, count):
        """Return a list concatenating self with itself ``count`` times."""
        return LazyConcatenation([self] * count)
    _MAX_REPR_SIZE = 60

    def __repr__(self):
        """
        Return a string representation for this corpus view that is
        similar to a list's representation; but if it would be more
        than 60 characters long, it is truncated.
        """
        pieces = []
        length = 5
        for elt in self:
            pieces.append(repr(elt))
            length += len(pieces[-1]) + 2
            if length > self._MAX_REPR_SIZE and len(pieces) > 2:
                return '[%s, ...]' % ', '.join(pieces[:-1])
        return '[%s]' % ', '.join(pieces)

    def __eq__(self, other):
        return type(self) == type(other) and list(self) == list(other)

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if type(other) != type(self):
            raise_unorderable_types('<', self, other)
        return list(self) < list(other)

    def __hash__(self):
        """
        :raise ValueError: Corpus view objects are unhashable.
        """
        raise ValueError('%s objects are unhashable' % self.__class__.__name__)