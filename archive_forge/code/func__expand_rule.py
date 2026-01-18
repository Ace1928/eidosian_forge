from collections import Counter, defaultdict
from typing import List, Dict, Iterator, FrozenSet, Set
from ..utils import bfs, fzset, classify
from ..exceptions import GrammarError
from ..grammar import Rule, Terminal, NonTerminal, Symbol
from ..common import ParserConf
def _expand_rule(rule: NonTerminal) -> Iterator[NonTerminal]:
    assert not rule.is_term, rule
    for r in rules_by_origin[rule]:
        init_ptr = RulePtr(r, 0)
        init_ptrs.add(init_ptr)
        if r.expansion:
            new_r = init_ptr.next
            if not new_r.is_term:
                assert isinstance(new_r, NonTerminal)
                yield new_r