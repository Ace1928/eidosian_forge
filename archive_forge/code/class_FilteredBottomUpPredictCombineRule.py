import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
class FilteredBottomUpPredictCombineRule(BottomUpPredictCombineRule):

    def apply(self, chart, grammar, edge):
        if edge.is_incomplete():
            return
        end = edge.end()
        nexttoken = end < chart.num_leaves() and chart.leaf(end)
        for prod in grammar.productions(rhs=edge.lhs()):
            if _bottomup_filter(grammar, nexttoken, prod.rhs()):
                new_edge = TreeEdge(edge.span(), prod.lhs(), prod.rhs(), 1)
                if chart.insert(new_edge, (edge,)):
                    yield new_edge