import re
from collections import defaultdict
from . import Tree, Token
from .common import ParserConf
from .parsers import earley
from .grammar import Rule, Terminal, NonTerminal
def _best_from_group(seq, group_key, cmp_key):
    d = {}
    for item in seq:
        key = group_key(item)
        if key in d:
            v1 = cmp_key(item)
            v2 = cmp_key(d[key])
            if v2 > v1:
                d[key] = item
        else:
            d[key] = item
    return list(d.values())