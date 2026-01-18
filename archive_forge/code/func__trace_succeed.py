from nltk.grammar import Nonterminal
from nltk.parse.api import ParserI
from nltk.tree import ImmutableTree, Tree
def _trace_succeed(self, tree, frontier):
    if self._trace > 2:
        print('GOOD PARSE:')
    if self._trace == 1:
        print('Found a parse:\n%s' % tree)
    if self._trace > 1:
        self._trace_tree(tree, frontier, '+')