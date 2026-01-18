import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
def _find_loops(self):
    """
        Find the loops defined by the graph's back edges.
        """
    bodies = {}
    for src, dest in self._back_edges:
        header = dest
        body = set([header])
        queue = [src]
        while queue:
            n = queue.pop()
            if n not in body:
                body.add(n)
                queue.extend(self._preds[n])
        if header in bodies:
            bodies[header].update(body)
        else:
            bodies[header] = body
    loops = {}
    for header, body in bodies.items():
        entries = set()
        exits = set()
        for n in body:
            entries.update(self._preds[n] - body)
            exits.update(self._succs[n] - body)
        loop = Loop(header=header, body=body, entries=entries, exits=exits)
        loops[header] = loop
    return loops