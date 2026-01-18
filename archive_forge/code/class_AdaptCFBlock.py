import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
class AdaptCFBlock(object):

    def __init__(self, blockinfo, offset):
        self.offset = offset
        self.body = tuple((i for i, _ in blockinfo.insts))