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
class Raise(Terminator):
    is_exit = True

    def __init__(self, exception, loc):
        assert exception is None or isinstance(exception, Var)
        assert isinstance(loc, Loc)
        self.exception = exception
        self.loc = loc

    def __str__(self):
        return 'raise %s' % self.exception

    def get_targets(self):
        return []