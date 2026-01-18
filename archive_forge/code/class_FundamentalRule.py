import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
class FundamentalRule(AbstractChartRule):
    """
    A rule that joins two adjacent edges to form a single combined
    edge.  In particular, this rule specifies that any pair of edges

    - ``[A -> alpha \\* B beta][i:j]``
    - ``[B -> gamma \\*][j:k]``

    licenses the edge:

    - ``[A -> alpha B * beta][i:j]``
    """
    NUM_EDGES = 2

    def apply(self, chart, grammar, left_edge, right_edge):
        if not (left_edge.is_incomplete() and right_edge.is_complete() and (left_edge.end() == right_edge.start()) and (left_edge.nextsym() == right_edge.lhs())):
            return
        new_edge = left_edge.move_dot_forward(right_edge.end())
        if chart.insert_with_backpointer(new_edge, left_edge, right_edge):
            yield new_edge