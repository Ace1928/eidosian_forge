from nltk.grammar import Nonterminal
from nltk.parse.api import ParserI
from nltk.tree import ImmutableTree, Tree
def _trace_backtrack(self, tree, frontier, toks=None):
    if self._trace > 2:
        if toks:
            print('Backtrack: %r match failed' % toks[0])
        else:
            print('Backtrack')