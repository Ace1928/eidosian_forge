import itertools
from nltk.ccg.combinator import *
from nltk.ccg.combinator import (
from nltk.ccg.lexicon import Token, fromstring
from nltk.ccg.logic import *
from nltk.parse import ParserI
from nltk.parse.chart import AbstractChartRule, Chart, EdgeI
from nltk.sem.logic import *
from nltk.tree import Tree
class CCGLeafEdge(EdgeI):
    """
    Class representing leaf edges in a CCG derivation.
    """

    def __init__(self, pos, token, leaf):
        self._pos = pos
        self._token = token
        self._leaf = leaf
        self._comparison_key = (pos, token.categ(), leaf)

    def lhs(self):
        return self._token.categ()

    def span(self):
        return (self._pos, self._pos + 1)

    def start(self):
        return self._pos

    def end(self):
        return self._pos + 1

    def length(self):
        return 1

    def rhs(self):
        return self._leaf

    def dot(self):
        return 0

    def is_complete(self):
        return True

    def is_incomplete(self):
        return False

    def nextsym(self):
        return None

    def token(self):
        return self._token

    def categ(self):
        return self._token.categ()

    def leaf(self):
        return self._leaf