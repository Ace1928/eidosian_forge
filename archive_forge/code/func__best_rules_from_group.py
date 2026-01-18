import re
from collections import defaultdict
from . import Tree, Token
from .common import ParserConf
from .parsers import earley
from .grammar import Rule, Terminal, NonTerminal
def _best_rules_from_group(rules):
    rules = _best_from_group(rules, lambda r: r, lambda r: -len(r.expansion))
    rules.sort(key=lambda r: len(r.expansion))
    return rules