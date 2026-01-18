import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def _dup_topx(self, state, inst, count):
    orig = [state.pop() for _ in range(count)]
    orig.reverse()
    duped = [state.make_temp() for _ in range(count)]
    state.append(inst, orig=orig, duped=duped)
    for val in orig:
        state.push(val)
    for val in duped:
        state.push(val)