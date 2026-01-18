from collections import defaultdict
import itertools
from ..exceptions import ParseError
from ..lexer import Token
from ..tree import Tree
from ..grammar import Terminal as T, NonTerminal as NT, Symbol
class Grammar:
    """Context-free grammar."""

    def __init__(self, rules):
        self.rules = frozenset(rules)

    def __eq__(self, other):
        return self.rules == other.rules

    def __str__(self):
        return '\n' + '\n'.join(sorted((repr(x) for x in self.rules))) + '\n'

    def __repr__(self):
        return str(self)