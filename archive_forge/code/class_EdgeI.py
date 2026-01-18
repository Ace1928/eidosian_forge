import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
@total_ordering
class EdgeI:
    """
    A hypothesis about the structure of part of a sentence.
    Each edge records the fact that a structure is (partially)
    consistent with the sentence.  An edge contains:

    - A span, indicating what part of the sentence is
      consistent with the hypothesized structure.
    - A left-hand side, specifying what kind of structure is
      hypothesized.
    - A right-hand side, specifying the contents of the
      hypothesized structure.
    - A dot position, indicating how much of the hypothesized
      structure is consistent with the sentence.

    Every edge is either complete or incomplete:

    - An edge is complete if its structure is fully consistent
      with the sentence.
    - An edge is incomplete if its structure is partially
      consistent with the sentence.  For every incomplete edge, the
      span specifies a possible prefix for the edge's structure.

    There are two kinds of edge:

    - A ``TreeEdge`` records which trees have been found to
      be (partially) consistent with the text.
    - A ``LeafEdge`` records the tokens occurring in the text.

    The ``EdgeI`` interface provides a common interface to both types
    of edge, allowing chart parsers to treat them in a uniform manner.
    """

    def __init__(self):
        if self.__class__ == EdgeI:
            raise TypeError('Edge is an abstract interface')

    def span(self):
        """
        Return a tuple ``(s, e)``, where ``tokens[s:e]`` is the
        portion of the sentence that is consistent with this
        edge's structure.

        :rtype: tuple(int, int)
        """
        raise NotImplementedError()

    def start(self):
        """
        Return the start index of this edge's span.

        :rtype: int
        """
        raise NotImplementedError()

    def end(self):
        """
        Return the end index of this edge's span.

        :rtype: int
        """
        raise NotImplementedError()

    def length(self):
        """
        Return the length of this edge's span.

        :rtype: int
        """
        raise NotImplementedError()

    def lhs(self):
        """
        Return this edge's left-hand side, which specifies what kind
        of structure is hypothesized by this edge.

        :see: ``TreeEdge`` and ``LeafEdge`` for a description of
            the left-hand side values for each edge type.
        """
        raise NotImplementedError()

    def rhs(self):
        """
        Return this edge's right-hand side, which specifies
        the content of the structure hypothesized by this edge.

        :see: ``TreeEdge`` and ``LeafEdge`` for a description of
            the right-hand side values for each edge type.
        """
        raise NotImplementedError()

    def dot(self):
        """
        Return this edge's dot position, which indicates how much of
        the hypothesized structure is consistent with the
        sentence.  In particular, ``self.rhs[:dot]`` is consistent
        with ``tokens[self.start():self.end()]``.

        :rtype: int
        """
        raise NotImplementedError()

    def nextsym(self):
        """
        Return the element of this edge's right-hand side that
        immediately follows its dot.

        :rtype: Nonterminal or terminal or None
        """
        raise NotImplementedError()

    def is_complete(self):
        """
        Return True if this edge's structure is fully consistent
        with the text.

        :rtype: bool
        """
        raise NotImplementedError()

    def is_incomplete(self):
        """
        Return True if this edge's structure is partially consistent
        with the text.

        :rtype: bool
        """
        raise NotImplementedError()

    def __eq__(self, other):
        return self.__class__ is other.__class__ and self._comparison_key == other._comparison_key

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if not isinstance(other, EdgeI):
            raise_unorderable_types('<', self, other)
        if self.__class__ is other.__class__:
            return self._comparison_key < other._comparison_key
        else:
            return self.__class__.__name__ < other.__class__.__name__

    def __hash__(self):
        try:
            return self._hash
        except AttributeError:
            self._hash = hash(self._comparison_key)
            return self._hash