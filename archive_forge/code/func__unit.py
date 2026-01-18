from collections import defaultdict
import itertools
from ..exceptions import ParseError
from ..lexer import Token
from ..tree import Tree
from ..grammar import Terminal as T, NonTerminal as NT, Symbol
def _unit(g):
    """Applies the UNIT rule to 'g' (see top comment)."""
    nt_unit_rule = get_any_nt_unit_rule(g)
    while nt_unit_rule:
        g = _remove_unit_rule(g, nt_unit_rule)
        nt_unit_rule = get_any_nt_unit_rule(g)
    return g