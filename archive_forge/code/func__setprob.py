import random
from functools import reduce
from nltk.grammar import PCFG, Nonterminal
from nltk.parse.api import ParserI
from nltk.parse.chart import AbstractChartRule, Chart, LeafEdge, TreeEdge
from nltk.tree import ProbabilisticTree, Tree
def _setprob(self, tree, prod_probs):
    if tree.prob() is not None:
        return
    lhs = Nonterminal(tree.label())
    rhs = []
    for child in tree:
        if isinstance(child, Tree):
            rhs.append(Nonterminal(child.label()))
        else:
            rhs.append(child)
    prob = prod_probs[lhs, tuple(rhs)]
    for child in tree:
        if isinstance(child, Tree):
            self._setprob(child, prod_probs)
            prob *= child.prob()
    tree.set_prob(prob)