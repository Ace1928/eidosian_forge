from collections import defaultdict
import itertools
from ..exceptions import ParseError
from ..lexer import Token
from ..tree import Tree
from ..grammar import Terminal as T, NonTerminal as NT, Symbol
def _term(g):
    """Applies the TERM rule on 'g' (see top comment)."""
    all_t = {x for rule in g.rules for x in rule.rhs if isinstance(x, T)}
    t_rules = {t: Rule(NT('__T_%s' % str(t)), [t], weight=0, alias='Term') for t in all_t}
    new_rules = []
    for rule in g.rules:
        if len(rule.rhs) > 1 and any((isinstance(x, T) for x in rule.rhs)):
            new_rhs = [t_rules[x].lhs if isinstance(x, T) else x for x in rule.rhs]
            new_rules.append(Rule(rule.lhs, new_rhs, weight=rule.weight, alias=rule.alias))
            new_rules.extend((v for k, v in t_rules.items() if k in rule.rhs))
        else:
            new_rules.append(rule)
    return Grammar(new_rules)