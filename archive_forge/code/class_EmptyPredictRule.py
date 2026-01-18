import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
class EmptyPredictRule(AbstractChartRule):
    """
    A rule that inserts all empty productions as passive edges,
    in every position in the chart.
    """
    NUM_EDGES = 0

    def apply(self, chart, grammar):
        for prod in grammar.productions(empty=True):
            for index in range(chart.num_leaves() + 1):
                new_edge = TreeEdge.from_production(prod, index)
                if chart.insert(new_edge, ()):
                    yield new_edge