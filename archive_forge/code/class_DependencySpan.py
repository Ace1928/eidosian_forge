from collections import defaultdict
from functools import total_ordering
from itertools import chain
from nltk.grammar import (
from nltk.internals import raise_unorderable_types
from nltk.parse.dependencygraph import DependencyGraph
@total_ordering
class DependencySpan:
    """
    A contiguous span over some part of the input string representing
    dependency (head -> modifier) relationships amongst words.  An atomic
    span corresponds to only one word so it isn't a 'span' in the conventional
    sense, as its _start_index = _end_index = _head_index for concatenation
    purposes.  All other spans are assumed to have arcs between all nodes
    within the start and end indexes of the span, and one head index corresponding
    to the head word for the entire span.  This is the same as the root node if
    the dependency structure were depicted as a graph.
    """

    def __init__(self, start_index, end_index, head_index, arcs, tags):
        self._start_index = start_index
        self._end_index = end_index
        self._head_index = head_index
        self._arcs = arcs
        self._tags = tags
        self._comparison_key = (start_index, end_index, head_index, tuple(arcs))
        self._hash = hash(self._comparison_key)

    def head_index(self):
        """
        :return: An value indexing the head of the entire ``DependencySpan``.
        :rtype: int
        """
        return self._head_index

    def __repr__(self):
        """
        :return: A concise string representatino of the ``DependencySpan``.
        :rtype: str.
        """
        return 'Span %d-%d; Head Index: %d' % (self._start_index, self._end_index, self._head_index)

    def __str__(self):
        """
        :return: A verbose string representation of the ``DependencySpan``.
        :rtype: str
        """
        str = 'Span %d-%d; Head Index: %d' % (self._start_index, self._end_index, self._head_index)
        for i in range(len(self._arcs)):
            str += '\n%d <- %d, %s' % (i, self._arcs[i], self._tags[i])
        return str

    def __eq__(self, other):
        return type(self) == type(other) and self._comparison_key == other._comparison_key

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if not isinstance(other, DependencySpan):
            raise_unorderable_types('<', self, other)
        return self._comparison_key < other._comparison_key

    def __hash__(self):
        """
        :return: The hash value of this ``DependencySpan``.
        """
        return self._hash