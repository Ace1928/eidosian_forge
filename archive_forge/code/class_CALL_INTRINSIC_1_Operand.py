import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
class CALL_INTRINSIC_1_Operand(Enum):
    INTRINSIC_STOPITERATION_ERROR = 3
    UNARY_POSITIVE = 5
    INTRINSIC_LIST_TO_TUPLE = 6