import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def _op_JUMP_IF_OR_POP(self, state, inst):
    pred = state.get_tos()
    state.append(inst, pred=pred)
    state.fork(pc=inst.next, npop=1)
    state.fork(pc=inst.get_jump_target())