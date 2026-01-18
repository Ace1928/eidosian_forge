import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
@functools.cached_property
def _in_loops(self):
    return self._find_in_loops()