import itertools
from nltk.ccg.combinator import *
from nltk.ccg.combinator import (
from nltk.ccg.lexicon import Token, fromstring
from nltk.ccg.logic import *
from nltk.parse import ParserI
from nltk.parse.chart import AbstractChartRule, Chart, EdgeI
from nltk.sem.logic import *
from nltk.tree import Tree
class CCGChart(Chart):

    def __init__(self, tokens):
        Chart.__init__(self, tokens)

    def _trees(self, edge, complete, memo, tree_class):
        assert complete, 'CCGChart cannot build incomplete trees'
        if edge in memo:
            return memo[edge]
        if isinstance(edge, CCGLeafEdge):
            word = tree_class(edge.token(), [self._tokens[edge.start()]])
            leaf = tree_class((edge.token(), 'Leaf'), [word])
            memo[edge] = [leaf]
            return [leaf]
        memo[edge] = []
        trees = []
        for cpl in self.child_pointer_lists(edge):
            child_choices = [self._trees(cp, complete, memo, tree_class) for cp in cpl]
            for children in itertools.product(*child_choices):
                lhs = (Token(self._tokens[edge.start():edge.end()], edge.lhs(), compute_semantics(children, edge)), str(edge.rule()))
                trees.append(tree_class(lhs, children))
        memo[edge] = trees
        return trees