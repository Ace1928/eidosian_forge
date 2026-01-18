import random
from functools import reduce
from nltk.grammar import PCFG, Nonterminal
from nltk.parse.api import ParserI
from nltk.parse.chart import AbstractChartRule, Chart, LeafEdge, TreeEdge
from nltk.tree import ProbabilisticTree, Tree
class UnsortedChartParser(BottomUpProbabilisticChartParser):
    """
    A bottom-up parser for ``PCFG`` grammars that tries edges in whatever order.
    """

    def sort_queue(self, queue, chart):
        return