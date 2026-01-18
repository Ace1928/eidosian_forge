from collections import defaultdict
import copy
import itertools
import os
import linecache
import pprint
import re
import sys
import operator
from types import FunctionType, BuiltinFunctionType
from functools import total_ordering
from io import StringIO
from numba.core import errors, config
from numba.core.utils import (BINOPS_TO_OPERATORS, INPLACE_BINOPS_TO_OPERATORS,
from numba.core.errors import (NotDefinedError, RedefinedError,
from numba.core import consts
class Terminator(Stmt):
    """
    IR statements that are terminators: the last statement in a block.
    A terminator must either:
    - exit the function
    - jump to a block

    All subclass of Terminator must override `.get_targets()` to return a list
    of jump targets.
    """
    is_terminator = True

    def get_targets(self):
        raise NotImplementedError(type(self))