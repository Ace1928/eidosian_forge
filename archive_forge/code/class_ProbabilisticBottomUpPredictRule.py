import random
from functools import reduce
from nltk.grammar import PCFG, Nonterminal
from nltk.parse.api import ParserI
from nltk.parse.chart import AbstractChartRule, Chart, LeafEdge, TreeEdge
from nltk.tree import ProbabilisticTree, Tree
class ProbabilisticBottomUpPredictRule(AbstractChartRule):
    NUM_EDGES = 1

    def apply(self, chart, grammar, edge):
        if edge.is_incomplete():
            return
        for prod in grammar.productions():
            if edge.lhs() == prod.rhs()[0]:
                new_edge = ProbabilisticTreeEdge.from_production(prod, edge.start(), prod.prob())
                if chart.insert(new_edge, ()):
                    yield new_edge