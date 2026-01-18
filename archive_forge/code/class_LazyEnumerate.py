import bisect
from collections import Counter, defaultdict, deque
from functools import total_ordering
from itertools import chain, islice
from nltk.internals import raise_unorderable_types, slice_bounds
class LazyEnumerate(LazyZip):
    """
    A lazy sequence whose elements are tuples, each containing a count (from
    zero) and a value yielded by underlying sequence.  ``LazyEnumerate`` is
    useful for obtaining an indexed list. The tuples are constructed lazily
    -- i.e., when you read a value from the list, ``LazyEnumerate`` will
    calculate that value by forming a tuple from the count of the i-th
    element and the i-th element of the underlying sequence.

    ``LazyEnumerate`` is essentially a lazy version of the Python primitive
    function ``enumerate``.  In particular, the following two expressions are
    equivalent:

        >>> from nltk.collections import LazyEnumerate
        >>> sequence = ['first', 'second', 'third']
        >>> list(enumerate(sequence))
        [(0, 'first'), (1, 'second'), (2, 'third')]
        >>> list(LazyEnumerate(sequence))
        [(0, 'first'), (1, 'second'), (2, 'third')]

    Lazy enumerations can be useful for conserving memory in cases where the
    argument sequences are particularly long.

    A typical example of a use case for this class is obtaining an indexed
    list for a long sequence of values.  By constructing tuples lazily and
    avoiding the creation of an additional long sequence, memory usage can be
    significantly reduced.
    """

    def __init__(self, lst):
        """
        :param lst: the underlying list
        :type lst: list
        """
        LazyZip.__init__(self, range(len(lst)), lst)