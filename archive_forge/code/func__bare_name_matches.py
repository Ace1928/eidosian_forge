from typing import (
from blib2to3.pgen2.grammar import Grammar
import sys
from io import StringIO
def _bare_name_matches(self, nodes) -> Tuple[int, _Results]:
    """Special optimized matcher for bare_name."""
    count = 0
    r = {}
    done = False
    max = len(nodes)
    while not done and count < max:
        done = True
        for leaf in self.content:
            if leaf[0].match(nodes[count], r):
                count += 1
                done = False
                break
    assert self.name is not None
    r[self.name] = nodes[:count]
    return (count, r)