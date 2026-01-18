import itertools
import sys
from nltk.grammar import Nonterminal
def _generate_one(grammar, item, depth):
    if depth > 0:
        if isinstance(item, Nonterminal):
            for prod in grammar.productions(lhs=item):
                yield from _generate_all(grammar, prod.rhs(), depth - 1)
        else:
            yield [item]