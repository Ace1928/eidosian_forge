import random
from functools import reduce
from nltk.grammar import PCFG, Nonterminal
from nltk.parse.api import ParserI
from nltk.parse.chart import AbstractChartRule, Chart, LeafEdge, TreeEdge
from nltk.tree import ProbabilisticTree, Tree
class ProbabilisticTreeEdge(TreeEdge):

    def __init__(self, prob, *args, **kwargs):
        TreeEdge.__init__(self, *args, **kwargs)
        self._prob = prob
        self._comparison_key = (self._comparison_key, prob)

    def prob(self):
        return self._prob

    @staticmethod
    def from_production(production, index, p):
        return ProbabilisticTreeEdge(p, (index, index), production.lhs(), production.rhs(), 0)