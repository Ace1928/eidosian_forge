import re
from nltk.grammar import Nonterminal, Production
from nltk.internals import deprecated
def _pformat_flat(self, nodesep, parens, quotes):
    childstrs = []
    for child in self:
        if isinstance(child, Tree):
            childstrs.append(child._pformat_flat(nodesep, parens, quotes))
        elif isinstance(child, tuple):
            childstrs.append('/'.join(child))
        elif isinstance(child, str) and (not quotes):
            childstrs.append('%s' % child)
        else:
            childstrs.append(repr(child))
    if isinstance(self._label, str):
        return '{}{}{} {}{}'.format(parens[0], self._label, nodesep, ' '.join(childstrs), parens[1])
    else:
        return '{}{}{} {}{}'.format(parens[0], repr(self._label), nodesep, ' '.join(childstrs), parens[1])