import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
class AdaptDFA(object):
    """Adapt Flow to the old DFA class expected by Interpreter
    """

    def __init__(self, flow):
        self._flow = flow

    @property
    def infos(self):
        return self._flow.block_infos