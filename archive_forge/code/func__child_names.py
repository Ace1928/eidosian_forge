import re
from nltk.grammar import Nonterminal, Production
from nltk.internals import deprecated
def _child_names(tree):
    names = []
    for child in tree:
        if isinstance(child, Tree):
            names.append(Nonterminal(child._label))
        else:
            names.append(child)
    return names