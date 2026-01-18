import re
from collections import defaultdict
from . import Tree, Token
from .common import ParserConf
from .parsers import earley
from .grammar import Rule, Terminal, NonTerminal
def _build_recons_rules(self, rules):
    """Convert tree-parsing/construction rules to tree-matching rules"""
    expand1s = {r.origin for r in rules if r.options.expand1}
    aliases = defaultdict(list)
    for r in rules:
        if r.alias:
            aliases[r.origin].append(r.alias)
    rule_names = {r.origin for r in rules}
    nonterminals = {sym for sym in rule_names if sym.name.startswith('_') or sym in expand1s or sym in aliases}
    seen = set()
    for r in rules:
        recons_exp = [sym if sym in nonterminals else Terminal(sym.name) for sym in r.expansion if not is_discarded_terminal(sym)]
        if recons_exp == [r.origin] and r.alias is None:
            continue
        sym = NonTerminal(r.alias) if r.alias else r.origin
        rule = make_recons_rule(sym, recons_exp, r.expansion)
        if sym in expand1s and len(recons_exp) != 1:
            self.rules_for_root[sym.name].append(rule)
            if sym.name not in seen:
                yield make_recons_rule_to_term(sym, sym)
                seen.add(sym.name)
        elif sym.name.startswith('_') or sym in expand1s:
            yield rule
        else:
            self.rules_for_root[sym.name].append(rule)
    for origin, rule_aliases in aliases.items():
        for alias in rule_aliases:
            yield make_recons_rule_to_term(origin, NonTerminal(alias))
        yield make_recons_rule_to_term(origin, origin)