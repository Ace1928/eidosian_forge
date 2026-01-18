from nltk.grammar import Nonterminal
from nltk.parse.api import ParserI
from nltk.tree import ImmutableTree, Tree
def _trace_tree(self, tree, frontier, operation):
    """
        Print trace output displaying the parser's current state.

        :param operation: A character identifying the operation that
            generated the current state.
        :rtype: None
        """
    if self._trace == 2:
        print('  %c [' % operation, end=' ')
    else:
        print('    [', end=' ')
    if len(frontier) > 0:
        self._trace_fringe(tree, frontier[0])
    else:
        self._trace_fringe(tree)
    print(']')