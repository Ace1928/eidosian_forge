import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
class BottomUpPredictCombineRule(BottomUpPredictRule):
    """
    A rule licensing any edge corresponding to a production whose
    right-hand side begins with a complete edge's left-hand side.  In
    particular, this rule specifies that ``[A -> alpha \\*]``
    licenses the edge ``[B -> A \\* beta]`` for each grammar
    production ``B -> A beta``.

    :note: This is like ``BottomUpPredictRule``, but it also applies
        the ``FundamentalRule`` to the resulting edge.
    """
    NUM_EDGES = 1

    def apply(self, chart, grammar, edge):
        if edge.is_incomplete():
            return
        for prod in grammar.productions(rhs=edge.lhs()):
            new_edge = TreeEdge(edge.span(), prod.lhs(), prod.rhs(), 1)
            if chart.insert(new_edge, (edge,)):
                yield new_edge