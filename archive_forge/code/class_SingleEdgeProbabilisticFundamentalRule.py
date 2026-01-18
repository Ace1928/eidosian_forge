import random
from functools import reduce
from nltk.grammar import PCFG, Nonterminal
from nltk.parse.api import ParserI
from nltk.parse.chart import AbstractChartRule, Chart, LeafEdge, TreeEdge
from nltk.tree import ProbabilisticTree, Tree
class SingleEdgeProbabilisticFundamentalRule(AbstractChartRule):
    NUM_EDGES = 1
    _fundamental_rule = ProbabilisticFundamentalRule()

    def apply(self, chart, grammar, edge1):
        fr = self._fundamental_rule
        if edge1.is_incomplete():
            for edge2 in chart.select(start=edge1.end(), is_complete=True, lhs=edge1.nextsym()):
                yield from fr.apply(chart, grammar, edge1, edge2)
        else:
            for edge2 in chart.select(end=edge1.start(), is_complete=False, nextsym=edge1.lhs()):
                yield from fr.apply(chart, grammar, edge2, edge1)

    def __str__(self):
        return 'Fundamental Rule'