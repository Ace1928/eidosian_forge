from collections import defaultdict
import itertools
from ..exceptions import ParseError
from ..lexer import Token
from ..tree import Tree
from ..grammar import Terminal as T, NonTerminal as NT, Symbol
def _bin(g):
    """Applies the BIN rule to 'g' (see top comment)."""
    new_rules = []
    for rule in g.rules:
        if len(rule.rhs) > 2:
            new_rules += _split(rule)
        else:
            new_rules.append(rule)
    return Grammar(new_rules)