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
def _rec_list_vars(self, val):
    """
        A recursive helper used to implement list_vars() in subclasses.
        """
    if isinstance(val, Var):
        return [val]
    elif isinstance(val, Inst):
        return val.list_vars()
    elif isinstance(val, (list, tuple)):
        lst = []
        for v in val:
            lst.extend(self._rec_list_vars(v))
        return lst
    elif isinstance(val, dict):
        lst = []
        for v in val.values():
            lst.extend(self._rec_list_vars(v))
        return lst
    else:
        return []