import random
from functools import reduce
from nltk.grammar import PCFG, Nonterminal
from nltk.parse.api import ParserI
from nltk.parse.chart import AbstractChartRule, Chart, LeafEdge, TreeEdge
from nltk.tree import ProbabilisticTree, Tree
class ProbabilisticFundamentalRule(AbstractChartRule):
    NUM_EDGES = 2

    def apply(self, chart, grammar, left_edge, right_edge):
        if not (left_edge.end() == right_edge.start() and left_edge.nextsym() == right_edge.lhs() and left_edge.is_incomplete() and right_edge.is_complete()):
            return
        p = left_edge.prob() * right_edge.prob()
        new_edge = ProbabilisticTreeEdge(p, span=(left_edge.start(), right_edge.end()), lhs=left_edge.lhs(), rhs=left_edge.rhs(), dot=left_edge.dot() + 1)
        changed_chart = False
        for cpl1 in chart.child_pointer_lists(left_edge):
            if chart.insert(new_edge, cpl1 + (right_edge,)):
                changed_chart = True
        if changed_chart:
            yield new_edge