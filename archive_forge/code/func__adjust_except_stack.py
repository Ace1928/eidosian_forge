import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def _adjust_except_stack(self, state):
    """
        Adjust stack when entering an exception handler to match expectation
        by the bytecode.
        """
    tryblk = state.get_top_block('TRY')
    state.pop_block_and_above(tryblk)
    nstack = state.stack_depth
    kwargs = {}
    expected_depth = tryblk['stack_depth']
    if nstack > expected_depth:
        kwargs['npop'] = nstack - expected_depth
    extra_stack = 1
    if tryblk['push_lasti']:
        extra_stack += 1
    kwargs['npush'] = extra_stack
    state.fork(pc=tryblk['end'], **kwargs)