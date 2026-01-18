from nltk.grammar import Nonterminal
from nltk.parse.api import ParserI
from nltk.tree import Tree
def _check_grammar(self):
    """
        Check to make sure that all of the CFG productions are
        potentially useful.  If any productions can never be used,
        then print a warning.

        :rtype: None
        """
    productions = self._grammar.productions()
    for i in range(len(productions)):
        for j in range(i + 1, len(productions)):
            rhs1 = productions[i].rhs()
            rhs2 = productions[j].rhs()
            if rhs1[:len(rhs2)] == rhs2:
                print('Warning: %r will never be used' % productions[i])