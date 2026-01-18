from collections import defaultdict
import itertools
from ..exceptions import ParseError
from ..lexer import Token
from ..tree import Tree
from ..grammar import Terminal as T, NonTerminal as NT, Symbol
class UnitSkipRule(Rule):
    """A rule that records NTs that were skipped during transformation."""

    def __init__(self, lhs, rhs, skipped_rules, weight, alias):
        super(UnitSkipRule, self).__init__(lhs, rhs, weight, alias)
        self.skipped_rules = skipped_rules

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.skipped_rules == other.skipped_rules
    __hash__ = Rule.__hash__