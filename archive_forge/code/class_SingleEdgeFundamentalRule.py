import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
class SingleEdgeFundamentalRule(FundamentalRule):
    """
    A rule that joins a given edge with adjacent edges in the chart,
    to form combined edges.  In particular, this rule specifies that
    either of the edges:

    - ``[A -> alpha \\* B beta][i:j]``
    - ``[B -> gamma \\*][j:k]``

    licenses the edge:

    - ``[A -> alpha B * beta][i:j]``

    if the other edge is already in the chart.

    :note: This is basically ``FundamentalRule``, with one edge left
        unspecified.
    """
    NUM_EDGES = 1

    def apply(self, chart, grammar, edge):
        if edge.is_incomplete():
            yield from self._apply_incomplete(chart, grammar, edge)
        else:
            yield from self._apply_complete(chart, grammar, edge)

    def _apply_complete(self, chart, grammar, right_edge):
        for left_edge in chart.select(end=right_edge.start(), is_complete=False, nextsym=right_edge.lhs()):
            new_edge = left_edge.move_dot_forward(right_edge.end())
            if chart.insert_with_backpointer(new_edge, left_edge, right_edge):
                yield new_edge

    def _apply_incomplete(self, chart, grammar, left_edge):
        for right_edge in chart.select(start=left_edge.end(), is_complete=True, lhs=left_edge.nextsym()):
            new_edge = left_edge.move_dot_forward(right_edge.end())
            if chart.insert_with_backpointer(new_edge, left_edge, right_edge):
                yield new_edge