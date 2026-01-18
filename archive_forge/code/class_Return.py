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
class Return(Terminator):
    """
    Return to caller.
    """
    is_exit = True

    def __init__(self, value, loc):
        assert isinstance(value, Var), type(value)
        assert isinstance(loc, Loc)
        self.value = value
        self.loc = loc

    def __str__(self):
        return 'return %s' % self.value

    def get_targets(self):
        return []