import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def _flatten_inst_regs(iterable):
    """Flatten an iterable of registers used in an instruction
    """
    for item in iterable:
        if isinstance(item, str):
            yield item
        elif isinstance(item, (tuple, list)):
            for x in _flatten_inst_regs(item):
                yield x