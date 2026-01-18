import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
def _find_in_loops(self):
    loops = self._loops
    in_loops = dict(((n, []) for n in self._nodes))
    for loop in sorted(loops.values(), key=lambda loop: len(loop.body)):
        for n in loop.body:
            in_loops[n].append(loop.header)
    return in_loops