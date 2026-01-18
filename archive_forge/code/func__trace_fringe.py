from nltk.grammar import Nonterminal
from nltk.parse.api import ParserI
from nltk.tree import ImmutableTree, Tree
def _trace_fringe(self, tree, treeloc=None):
    """
        Print trace output displaying the fringe of ``tree``.  The
        fringe of ``tree`` consists of all of its leaves and all of
        its childless subtrees.

        :rtype: None
        """
    if treeloc == ():
        print('*', end=' ')
    if isinstance(tree, Tree):
        if len(tree) == 0:
            print(repr(Nonterminal(tree.label())), end=' ')
        for i in range(len(tree)):
            if treeloc is not None and i == treeloc[0]:
                self._trace_fringe(tree[i], treeloc[1:])
            else:
                self._trace_fringe(tree[i])
    else:
        print(repr(tree), end=' ')