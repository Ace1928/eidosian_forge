from typing import (
from blib2to3.pgen2.grammar import Grammar
import sys
from io import StringIO
def _recursive_matches(self, nodes, count) -> Iterator[Tuple[int, _Results]]:
    """Helper to recursively yield the matches."""
    assert self.content is not None
    if count >= self.min:
        yield (0, {})
    if count < self.max:
        for alt in self.content:
            for c0, r0 in generate_matches(alt, nodes):
                for c1, r1 in self._recursive_matches(nodes[c0:], count + 1):
                    r = {}
                    r.update(r0)
                    r.update(r1)
                    yield (c0 + c1, r)