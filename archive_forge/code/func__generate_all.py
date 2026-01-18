import itertools
import sys
from nltk.grammar import Nonterminal
def _generate_all(grammar, items, depth):
    if items:
        try:
            for frag1 in _generate_one(grammar, items[0], depth):
                for frag2 in _generate_all(grammar, items[1:], depth):
                    yield (frag1 + frag2)
        except RecursionError as error:
            raise RuntimeError('The grammar has rule(s) that yield infinite recursion!') from error
    else:
        yield []