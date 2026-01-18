import itertools
from nltk.ccg.combinator import *
from nltk.ccg.combinator import (
from nltk.ccg.lexicon import Token, fromstring
from nltk.ccg.logic import *
from nltk.parse import ParserI
from nltk.parse.chart import AbstractChartRule, Chart, EdgeI
from nltk.sem.logic import *
from nltk.tree import Tree
class ForwardTypeRaiseRule(AbstractChartRule):
    """
    Class for applying forward type raising
    """
    NUMEDGES = 2

    def __init__(self):
        self._combinator = ForwardT

    def apply(self, chart, grammar, left_edge, right_edge):
        if not left_edge.end() == right_edge.start():
            return
        for res in self._combinator.combine(left_edge.categ(), right_edge.categ()):
            new_edge = CCGEdge(span=left_edge.span(), categ=res, rule=self._combinator)
            if chart.insert(new_edge, (left_edge,)):
                yield new_edge

    def __str__(self):
        return '%s' % self._combinator