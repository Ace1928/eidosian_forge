from typing import (
from blib2to3.pgen2.grammar import Grammar
import sys
from io import StringIO
def _iterative_matches(self, nodes) -> Iterator[Tuple[int, _Results]]:
    """Helper to iteratively yield the matches."""
    nodelen = len(nodes)
    if 0 >= self.min:
        yield (0, {})
    results = []
    for alt in self.content:
        for c, r in generate_matches(alt, nodes):
            yield (c, r)
            results.append((c, r))
    while results:
        new_results = []
        for c0, r0 in results:
            if c0 < nodelen and c0 <= self.max:
                for alt in self.content:
                    for c1, r1 in generate_matches(alt, nodes[c0:]):
                        if c1 > 0:
                            r = {}
                            r.update(r0)
                            r.update(r1)
                            yield (c0 + c1, r)
                            new_results.append((c0 + c1, r))
        results = new_results