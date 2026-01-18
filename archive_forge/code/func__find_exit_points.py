import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
def _find_exit_points(self):
    """
        Compute the graph's exit points.
        """
    exit_points = set()
    for n in self._nodes:
        if not self._succs.get(n):
            exit_points.add(n)
    return exit_points