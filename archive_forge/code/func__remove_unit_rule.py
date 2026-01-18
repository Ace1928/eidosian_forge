from collections import defaultdict
import itertools
from ..exceptions import ParseError
from ..lexer import Token
from ..tree import Tree
from ..grammar import Terminal as T, NonTerminal as NT, Symbol
def _remove_unit_rule(g, rule):
    """Removes 'rule' from 'g' without changing the language produced by 'g'."""
    new_rules = [x for x in g.rules if x != rule]
    refs = [x for x in g.rules if x.lhs == rule.rhs[0]]
    new_rules += [build_unit_skiprule(rule, ref) for ref in refs]
    return Grammar(new_rules)