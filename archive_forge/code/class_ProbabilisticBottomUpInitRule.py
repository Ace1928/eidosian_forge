import random
from functools import reduce
from nltk.grammar import PCFG, Nonterminal
from nltk.parse.api import ParserI
from nltk.parse.chart import AbstractChartRule, Chart, LeafEdge, TreeEdge
from nltk.tree import ProbabilisticTree, Tree
class ProbabilisticBottomUpInitRule(AbstractChartRule):
    NUM_EDGES = 0

    def apply(self, chart, grammar):
        for index in range(chart.num_leaves()):
            new_edge = ProbabilisticLeafEdge(chart.leaf(index), index)
            if chart.insert(new_edge, ()):
                yield new_edge