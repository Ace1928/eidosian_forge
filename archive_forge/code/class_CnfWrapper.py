from collections import defaultdict
import itertools
from ..exceptions import ParseError
from ..lexer import Token
from ..tree import Tree
from ..grammar import Terminal as T, NonTerminal as NT, Symbol
class CnfWrapper:
    """CNF wrapper for grammar.

  Validates that the input grammar is CNF and provides helper data structures.
  """

    def __init__(self, grammar):
        super(CnfWrapper, self).__init__()
        self.grammar = grammar
        self.rules = grammar.rules
        self.terminal_rules = defaultdict(list)
        self.nonterminal_rules = defaultdict(list)
        for r in self.rules:
            assert isinstance(r.lhs, NT), r
            if len(r.rhs) not in [1, 2]:
                raise ParseError("CYK doesn't support empty rules")
            if len(r.rhs) == 1 and isinstance(r.rhs[0], T):
                self.terminal_rules[r.rhs[0]].append(r)
            elif len(r.rhs) == 2 and all((isinstance(x, NT) for x in r.rhs)):
                self.nonterminal_rules[tuple(r.rhs)].append(r)
            else:
                assert False, r

    def __eq__(self, other):
        return self.grammar == other.grammar

    def __repr__(self):
        return repr(self.grammar)