from collections import defaultdict
from functools import total_ordering
from itertools import chain
from nltk.grammar import (
from nltk.internals import raise_unorderable_types
from nltk.parse.dependencygraph import DependencyGraph
class ChartCell:
    """
    A cell from the parse chart formed when performing the CYK algorithm.
    Each cell keeps track of its x and y coordinates (though this will probably
    be discarded), and a list of spans serving as the cell's entries.
    """

    def __init__(self, x, y):
        """
        :param x: This cell's x coordinate.
        :type x: int.
        :param y: This cell's y coordinate.
        :type y: int.
        """
        self._x = x
        self._y = y
        self._entries = set()

    def add(self, span):
        """
        Appends the given span to the list of spans
        representing the chart cell's entries.

        :param span: The span to add.
        :type span: DependencySpan
        """
        self._entries.add(span)

    def __str__(self):
        """
        :return: A verbose string representation of this ``ChartCell``.
        :rtype: str.
        """
        return 'CC[%d,%d]: %s' % (self._x, self._y, self._entries)

    def __repr__(self):
        """
        :return: A concise string representation of this ``ChartCell``.
        :rtype: str.
        """
        return '%s' % self