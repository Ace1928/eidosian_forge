import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
class ChartRuleI:
    """
    A rule that specifies what new edges are licensed by any given set
    of existing edges.  Each chart rule expects a fixed number of
    edges, as indicated by the class variable ``NUM_EDGES``.  In
    particular:

    - A chart rule with ``NUM_EDGES=0`` specifies what new edges are
      licensed, regardless of existing edges.
    - A chart rule with ``NUM_EDGES=1`` specifies what new edges are
      licensed by a single existing edge.
    - A chart rule with ``NUM_EDGES=2`` specifies what new edges are
      licensed by a pair of existing edges.

    :type NUM_EDGES: int
    :cvar NUM_EDGES: The number of existing edges that this rule uses
        to license new edges.  Typically, this number ranges from zero
        to two.
    """

    def apply(self, chart, grammar, *edges):
        """
        Return a generator that will add edges licensed by this rule
        and the given edges to the chart, one at a time.  Each
        time the generator is resumed, it will either add a new
        edge and yield that edge; or return.

        :type edges: list(EdgeI)
        :param edges: A set of existing edges.  The number of edges
            that should be passed to ``apply()`` is specified by the
            ``NUM_EDGES`` class variable.
        :rtype: iter(EdgeI)
        """
        raise NotImplementedError()

    def apply_everywhere(self, chart, grammar):
        """
        Return a generator that will add all edges licensed by
        this rule, given the edges that are currently in the
        chart, one at a time.  Each time the generator is resumed,
        it will either add a new edge and yield that edge; or return.

        :rtype: iter(EdgeI)
        """
        raise NotImplementedError()